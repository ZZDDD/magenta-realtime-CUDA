import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {
  Activity, CircleStop, Download, Gauge, Keyboard, Pause, Play,
  Plus, Radio, RefreshCw, RotateCcw, SlidersHorizontal, Trash2, Upload,
} from 'lucide-react';
import {
  PianoKeyboard, PromptSurface, calculateWeights,
  type ListenerNode, type PromptNode,
} from '@magenta-rt/common';
import {PcmPlayer} from './audio';

type Mode = 'mrt2' | 'jam' | 'collider' | 'advanced';
type MidiStatus = 'idle' | 'requesting' | 'connected' | 'empty' | 'unsupported' | 'denied' | 'error';
type ChunkSample = {sequence: number; revision: number; gapMs: number};
type Prompt = {text: string; weight: number; source?: string};
type Params = {
  temperature: number; top_k: number; cfg_musiccoca: number;
  cfg_notes: number; cfg_drums: number; drumless: boolean;
  onset_mode: boolean; unmask_width: number;
};
type Status = {
  model: string; device: number; stage: string; ready: boolean;
  frame_ms: number | null; p95_ms: number | null; error?: string | null;
  worker_id?: number;
};

const DEFAULT_PARAMS: Params = {
  temperature: 1.3, top_k: 40, cfg_musiccoca: 3, cfg_notes: 1,
  cfg_drums: 1, drumless: false, onset_mode: false, unmask_width: 0,
};
const PRESETS = [
  'disco funk', 'ambient pads with sub bass', 'jazz piano trio',
  'minimal techno', 'cinematic strings', 'lo-fi guitar groove',
  'Afrobeat brass ensemble', 'solo concert piano',
];
const COMPUTER_KEY_NOTES: Record<string, number> = {
  KeyA: 48, KeyW: 49, KeyS: 50, KeyE: 51, KeyD: 52,
  KeyF: 53, KeyT: 54, KeyG: 55, KeyY: 56, KeyH: 57,
  KeyU: 58, KeyJ: 59, KeyK: 60, KeyO: 61, KeyL: 62,
  KeyP: 63, Semicolon: 64,
};

function Metric({label, value}: {label: string; value: string}) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Slider({label, value, min, max, step, onChange}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (value: number) => void;
}) {
  return <label className="slider-row">
    <span>{label}<output>{value.toFixed(step < 1 ? 2 : 0)}</output></span>
    <input type="range" value={value} min={min} max={max} step={step}
      onChange={event => onChange(Number(event.target.value))} />
  </label>;
}

export default function App() {
  const [mode, setMode] = useState<Mode>('mrt2');
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState<Status>({
    model: 'mrt2_base', device: 0, stage: 'connecting', ready: false,
    frame_ms: null, p95_ms: null,
  });
  const [playing, setPlaying] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordingUrl, setRecordingUrl] = useState<string | null>(null);
  const [params, setParams] = useState<Params>(DEFAULT_PARAMS);
  const [prompts, setPrompts] = useState<Prompt[]>([
    {text: 'disco funk', weight: 1},
    {text: 'ambient pads with sub bass', weight: 0},
  ]);
  const [activeNotes, setActiveNotes] = useState<number[]>([]);
  const [audioStats, setAudioStats] = useState({queue: 0, underruns: 0, level: 0});
  const [computerKeyboardEnabled, setComputerKeyboardEnabled] = useState(true);
  const [midiStatus, setMidiStatus] = useState<MidiStatus>('idle');
  const [midiInputCount, setMidiInputCount] = useState(0);
  const [selectedNode, setSelectedNode] = useState<number | null>(null);
  const [listener, setListener] = useState<ListenerNode>({x: 400, y: 210});
  const [positions, setPositions] = useState([
    {x: 170, y: 120}, {x: 610, y: 130}, {x: 390, y: 330},
    {x: 650, y: 320}, {x: 260, y: 310}, {x: 430, y: 100},
  ]);
  const websocket = useRef<WebSocket | null>(null);
  const player = useRef<PcmPlayer | null>(null);
  const reconnectTimer = useRef<number>();
  const midiAccess = useRef<MIDIAccess | null>(null);
  const latestSequence = useRef(0);
  const chunkCanvas = useRef<HTMLCanvasElement | null>(null);
  const chunkHistory = useRef<ChunkSample[]>([]);
  const lastChunkArrival = useRef<number | null>(null);
  const heldComputerKeys = useRef(new Set<string>());
  const [chunkInfo, setChunkInfo] = useState<ChunkSample | null>(null);

  const resetChunkMonitor = useCallback(() => {
    chunkHistory.current = [];
    lastChunkArrival.current = null;
    setChunkInfo(null);
    const canvas = chunkCanvas.current;
    canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const drawChunk = useCallback((sample: ChunkSample) => {
    const canvas = chunkCanvas.current;
    if (!canvas) return;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const scale = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * scale) || canvas.height !== Math.round(height * scale)) {
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
    }
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(scale, 0, 0, scale, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#242828';
    context.fillRect(0, height - 1, width, 1);
    const barWidth = 3;
    const stride = 4;
    const visible = chunkHistory.current.slice(-Math.floor(width / stride));
    visible.forEach((chunk, index) => {
      const barHeight = Math.min(height - 3, Math.max(5, chunk.gapMs / 2));
      context.fillStyle = chunk.gapMs <= 48 ? '#8ddeaa' : chunk.gapMs <= 65 ? '#f1c75b' : '#f45b69';
      context.fillRect(index * stride, height - barHeight, barWidth, barHeight);
    });
  }, []);

  const send = useCallback((message: object) => {
    if (websocket.current?.readyState === WebSocket.OPEN) {
      websocket.current.send(JSON.stringify(message));
    }
  }, []);

  useEffect(() => {
    let disposed = false;
    const connect = () => {
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
      const socket = new WebSocket(`${protocol}://${location.host}/api/session`);
      socket.binaryType = 'arraybuffer';
      websocket.current = socket;
      socket.onopen = () => {
        latestSequence.current = 0;
        player.current?.clear();
        resetChunkMonitor();
        setConnected(true);
      };
      socket.onclose = () => {
        setConnected(false);
        if (!disposed) reconnectTimer.current = window.setTimeout(connect, 1500);
      };
      socket.onmessage = event => {
        if (event.data instanceof ArrayBuffer) {
          const view = new DataView(event.data);
          if (view.byteLength < 24 || view.getUint32(0, false) !== 0x4d525432) return;
          const sequence = view.getUint32(4, true);
          const revision = view.getUint32(8, true);
          const samples = view.getUint32(12, true);
          if (samples !== 1920 || view.byteLength !== 24 + samples * 4) return;
          if (sequence <= latestSequence.current) return;
          latestSequence.current = sequence;
          const arrived = performance.now();
          const gapMs = lastChunkArrival.current == null ? 40 : arrived - lastChunkArrival.current;
          lastChunkArrival.current = arrived;
          const sample = {sequence, revision, gapMs};
          chunkHistory.current.push(sample);
          if (chunkHistory.current.length > 96) chunkHistory.current.shift();
          drawChunk(sample);
          if (sequence % 5 === 0) setChunkInfo(sample);
          const pcm = new Int16Array(event.data, 24, samples * 2);
          player.current?.write(pcm, revision);
          return;
        }
        const message = JSON.parse(event.data);
        if (message.type === 'status' || message.type === 'metrics') {
          setStatus(previous => ({...previous, ...message}));
        } else if (message.type === 'recording') {
          setRecording(Boolean(message.recording));
          if (message.url) setRecordingUrl(message.url);
        } else if (message.type === 'error') {
          setStatus(previous => ({...previous, error: message.message}));
        }
      };
    };
    connect();
    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer.current);
      websocket.current?.close();
    };
  }, [drawChunk, resetChunkMonitor]);

  useEffect(() => {
    if (!connected) return;
    const timer = window.setTimeout(() => send({type: 'params', ...params}), 80);
    return () => window.clearTimeout(timer);
  }, [connected, params, send, status.worker_id]);

  useEffect(() => {
    if (!connected) return;
    const timer = window.setTimeout(() => send({type: 'prompts', prompts}), 400);
    return () => window.clearTimeout(timer);
  }, [connected, prompts, send, status.worker_id]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (!player.current) return;
      setAudioStats({
        queue: player.current.queuedFrames,
        underruns: player.current.underruns,
        level: player.current.level,
      });
    }, 250);
    return () => window.clearInterval(timer);
  }, []);

  const togglePlay = async () => {
    if (!status.ready) return;
    const next = !playing;
    if (next) {
      player.current ??= new PcmPlayer(2);
      player.current.setTargetFrames(status.model === 'mrt2_small' ? 1 : 2);
      await player.current.start();
    } else {
      player.current?.suspend();
      player.current?.clear();
    }
    setPlaying(next);
    send({type: 'transport', playing: next});
  };

  const changeModel = (model: string) => {
    setPlaying(false);
    player.current?.clear();
    latestSequence.current = 0;
    resetChunkMonitor();
    send({type: 'model', model});
    setStatus(previous => ({...previous, model, ready: false, stage: 'switching'}));
  };

  const note = useCallback((midi: number, on: boolean) => {
    setActiveNotes(previous => on
      ? Array.from(new Set([...previous, midi]))
      : previous.filter(value => value !== midi));
    send({type: 'midi', note: midi, on});
  }, [send]);

  useEffect(() => {
    const releaseComputerKeys = () => {
      heldComputerKeys.current.forEach(code => note(COMPUTER_KEY_NOTES[code], false));
      heldComputerKeys.current.clear();
    };
    if (!computerKeyboardEnabled || mode !== 'jam') {
      releaseComputerKeys();
      return;
    }
    const isTypingTarget = (target: EventTarget | null) => {
      const element = target instanceof HTMLElement ? target : null;
      return element?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(element?.tagName ?? '');
    };
    const keyDown = (event: KeyboardEvent) => {
      const midi = COMPUTER_KEY_NOTES[event.code];
      if (midi == null || event.repeat || isTypingTarget(event.target)) return;
      event.preventDefault();
      if (heldComputerKeys.current.has(event.code)) return;
      heldComputerKeys.current.add(event.code);
      note(midi, true);
    };
    const keyUp = (event: KeyboardEvent) => {
      if (!heldComputerKeys.current.delete(event.code)) return;
      event.preventDefault();
      note(COMPUTER_KEY_NOTES[event.code], false);
    };
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', releaseComputerKeys);
    return () => {
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', releaseComputerKeys);
      releaseComputerKeys();
    };
  }, [computerKeyboardEnabled, mode, note]);

  const enableMidi = async () => {
    if (!navigator.requestMIDIAccess) {
      setMidiStatus('unsupported');
      return;
    }
    setMidiStatus('requesting');
    try {
      const access = midiAccess.current ?? await navigator.requestMIDIAccess();
      midiAccess.current = access;
      const connectInputs = () => {
        let count = 0;
        access.inputs.forEach(input => {
          if (input.state !== 'connected') {
            input.onmidimessage = null;
            return;
          }
          count += 1;
          input.onmidimessage = event => {
            if (!event.data) return;
            const [command, midi, velocity] = event.data;
            if ((command & 0xf0) === 0x90 && velocity > 0) note(midi, true);
            if ((command & 0xf0) === 0x80 || ((command & 0xf0) === 0x90 && velocity === 0)) note(midi, false);
          };
        });
        setMidiInputCount(count);
        setMidiStatus(count > 0 ? 'connected' : 'empty');
      };
      connectInputs();
      access.onstatechange = connectInputs;
    } catch (error) {
      const name = error instanceof DOMException ? error.name : '';
      setMidiStatus(name === 'NotAllowedError' || name === 'SecurityError' ? 'denied' : 'error');
    }
  };

  const midiLabel = {
    idle: 'Enable browser MIDI',
    requesting: 'Requesting MIDI...',
    connected: `${midiInputCount} MIDI input${midiInputCount === 1 ? '' : 's'}`,
    empty: 'No MIDI inputs',
    unsupported: 'MIDI unsupported',
    denied: 'MIDI permission denied',
    error: 'MIDI unavailable',
  }[midiStatus];

  const uploadAudio = async (file: File) => {
    const form = new FormData();
    form.append('file', file);
    const response = await fetch('/api/prompts/audio', {method: 'POST', body: form});
    if (!response.ok) {
      const error = await response.text();
      setStatus(previous => ({...previous, error}));
      return;
    }
    const uploaded = await response.json() as {id: string; label: string};
    setPrompts(items => {
      const prompt = {text: uploaded.label, weight: 1, source: uploaded.id};
      return items.length < 6 ? [...items, prompt] : [...items.slice(0, 5), prompt];
    });
  };

  const nodes: PromptNode[] = useMemo(() => prompts.map((prompt, index) => ({
    id: index, x: positions[index]?.x ?? 400, y: positions[index]?.y ?? 210,
    label: prompt.text || `Prompt ${index + 1}`, colorIndex: index,
  })), [positions, prompts]);

  const updateSurfaceWeights = (nextListener: ListenerNode, nextNodes = nodes) => {
    const weights = calculateWeights(nextListener, nextNodes);
    setPrompts(previous => previous.map((prompt, index) => ({
      ...prompt, weight: weights[index] ?? prompt.weight,
    })));
  };

  const removePrompt = (index: number) => {
    setPrompts(items => items.filter((_, itemIndex) => itemIndex !== index));
    setPositions(items => items.filter((_, itemIndex) => itemIndex !== index));
    setSelectedNode(current => {
      if (current == null || current === index) return null;
      return current > index ? current - 1 : current;
    });
  };

  return <div className="app-shell">
    <header className="topbar">
      <div className="brand"><Radio size={20} /><strong>Magenta RT</strong><span>CUDA host</span></div>
      <nav className="mode-tabs" aria-label="Application mode">
        {(['mrt2', 'jam', 'collider', 'advanced'] as Mode[]).map(item =>
          <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>
            {item === 'mrt2' ? 'MRT2' : item[0].toUpperCase() + item.slice(1)}
          </button>)}
      </nav>
      <div className="connection"><span className={connected ? 'dot online' : 'dot'} />{status.stage}</div>
    </header>

    <section className="transport-band">
      <select value={status.model} onChange={event => changeModel(event.target.value)}
        disabled={status.stage === 'switching' || status.stage === 'loading'} aria-label="Model">
        <option value="mrt2_base">MRT2 Base · 2.4B</option>
        <option value="mrt2_small">MRT2 Small · 230M</option>
      </select>
      <button className="icon primary" onClick={togglePlay} disabled={!status.ready} title={playing ? 'Pause' : 'Play'}>
        {playing ? <Pause /> : <Play />}
      </button>
      <button className={`icon ${recording ? 'recording' : ''}`} onClick={() => send({type: 'recording', action: recording ? 'stop' : 'start'})}
        disabled={!playing} title={recording ? 'Stop recording' : 'Record'}><CircleStop /></button>
      <button className="icon" onClick={() => send({type: 'reset'})} title="Reset model state"><RotateCcw /></button>
      {recordingUrl && <a className="download" href={recordingUrl} download><Download size={17} /> WAV</a>}
      <div className="transport-spacer" />
      <div className="chunk-monitor" title="Realtime chunk arrivals">
        <canvas ref={chunkCanvas} />
        <div><span>Chunks</span><strong>{chunkInfo == null ? 'Waiting' : `#${chunkInfo.sequence} · ${chunkInfo.gapMs.toFixed(1)} ms`}</strong>
          <small>{chunkInfo == null ? '40 ms frames' : `control r${chunkInfo.revision}`}</small></div>
      </div>
      <Metric label="Frame" value={status.frame_ms == null ? '—' : `${status.frame_ms.toFixed(1)} ms`} />
      <Metric label="P95" value={status.p95_ms == null ? '—' : `${status.p95_ms.toFixed(1)} ms`} />
      <Metric label="Queue" value={`${audioStats.queue.toFixed(1)} fr`} />
      <Metric label="Underruns" value={String(audioStats.underruns)} />
      <div className="level" title="Output level"><i style={{transform: `scaleX(${Math.min(1, audioStats.level * 5)})`}} /></div>
    </section>

    {status.error && <div className="error-bar">{status.error}</div>}

    <main>
      {mode === 'mrt2' && <div className="workspace two-column">
        <section className="prompt-editor">
          <div className="section-heading"><div><span>STYLE MIX</span><h1>Prompt console</h1></div>
            <button className="icon" title="Add prompt" disabled={prompts.length >= 6}
              onClick={() => setPrompts(items => [...items, {text: '', weight: 0.5}])}><Plus /></button></div>
          <div className="prompt-list">
            {prompts.map((prompt, index) => <div className="prompt-row" key={index}>
              <b style={{background: `var(--prompt-${index % 6})`}}>{index + 1}</b>
              <input value={prompt.text} placeholder="Describe a musical style"
                onChange={event => setPrompts(items => items.map((item, i) => i === index ? {...item, text: event.target.value} : item))} />
              <input type="range" min="0" max="1" step="0.01" value={prompt.weight}
                onChange={event => setPrompts(items => items.map((item, i) => i === index ? {...item, weight: Number(event.target.value)} : item))} />
              <output>{Math.round(prompt.weight * 100)}%</output>
              <button className="icon small" title="Remove prompt" onClick={() => removePrompt(index)}><Trash2 /></button>
            </div>)}
          </div>
          <label className="upload-button"><Upload size={18} /> Audio prompt<input type="file" accept="audio/*" onChange={event => event.target.files?.[0] && uploadAudio(event.target.files[0])} /></label>
        </section>
        <section className="control-panel">
          <div className="section-heading"><div><span>GENERATION</span><h2>Live controls</h2></div><SlidersHorizontal /></div>
          <Slider label="Temperature" value={params.temperature} min={0} max={3} step={0.01} onChange={temperature => setParams({...params, temperature})} />
          <Slider label="Top-K" value={params.top_k} min={1} max={200} step={1} onChange={top_k => setParams({...params, top_k})} />
          <Slider label="Prompt strength" value={params.cfg_musiccoca} min={-1} max={7} step={0.1} onChange={cfg_musiccoca => setParams({...params, cfg_musiccoca})} />
          <div className="toggle-grid">
            <label><input type="checkbox" checked={params.drumless} onChange={event => setParams({...params, drumless: event.target.checked})} /> No drums</label>
            <label><input type="checkbox" checked={params.onset_mode} onChange={event => setParams({...params, onset_mode: event.target.checked})} /> Explicit onsets</label>
          </div>
        </section>
      </div>}

      {mode === 'jam' && <div className="jam-workspace">
        <section className="preset-strip">
          <span>STYLE PRESETS</span>
          {PRESETS.map(preset => <button key={preset} onClick={() => setPrompts([{text: preset, weight: 1}])}
            className={prompts[0]?.text === preset ? 'selected' : ''}>{preset}</button>)}
        </section>
        <section className="jam-controls">
          <Slider label="Chaos" value={params.temperature} min={0.2} max={2.5} step={0.01} onChange={temperature => setParams({...params, temperature})} />
          <Slider label="Style" value={params.cfg_musiccoca} min={0} max={7} step={0.1} onChange={cfg_musiccoca => setParams({...params, cfg_musiccoca})} />
          <Slider label="Notes" value={params.cfg_notes} min={0} max={7} step={0.1} onChange={cfg_notes => setParams({...params, cfg_notes})} />
        </section>
        <section className="keyboard-area">
          <div className="keyboard-heading"><div><Keyboard /><span>NOTE CONTROL</span></div>
            <div className="note-inputs">
              <label className={`computer-input ${computerKeyboardEnabled ? 'active' : ''}`}><input type="checkbox" checked={computerKeyboardEnabled}
                onChange={event => setComputerKeyboardEnabled(event.target.checked)} /> Computer keys</label>
              <button onClick={enableMidi} className={`midi-${midiStatus}`}
                disabled={midiStatus === 'requesting'} aria-live="polite">{midiLabel}</button>
            </div></div>
          <PianoKeyboard activeNotes={activeNotes} accentColor="#f45b69" startNote={48} endNote={76}
            keyboardMidiEnabled={computerKeyboardEnabled} onNoteOn={midi => note(midi, true)} onNoteOff={midi => note(midi, false)} showOctaveLabels />
        </section>
      </div>}

      {mode === 'collider' && <div className="collider-workspace">
        <PromptSurface prompts={nodes} listener={listener} selectedBallId={selectedNode}
          onPromptMove={(id, x, y) => {
            const next = positions.map((position, index) => index === id ? {x, y} : position);
            setPositions(next); updateSurfaceWeights(listener, nodes.map(node => node.id === id ? {...node, x, y} : node));
          }}
          onListenerMove={(x, y) => {const next = {x, y}; setListener(next); updateSurfaceWeights(next);}}
          onBallSelect={setSelectedNode} onPromptAdd={() => undefined}
          onPromptTextChange={(id, text) => setPrompts(items => items.map((item, index) => index === id ? {...item, text} : item))}
          onPromptDelete={removePrompt}
          physicsSpeed={1} onFirstThrow={() => undefined} isPlaying={playing}
          audioLevel={audioStats.level} collisions physicsEnabled active />
        <div className="collider-caption"><Activity /><span>Drag the listener through prompt space</span></div>
        <button className="icon collider-delete" title="Remove selected prompt"
          disabled={selectedNode == null} onClick={() => selectedNode != null && removePrompt(selectedNode)}><Trash2 /></button>
      </div>}

      {mode === 'advanced' && <div className="workspace advanced-grid">
        <section><div className="section-heading"><div><span>CONDITIONING</span><h2>Model controls</h2></div><Gauge /></div>
          <Slider label="Notes CFG" value={params.cfg_notes} min={-1} max={7} step={0.1} onChange={cfg_notes => setParams({...params, cfg_notes})} />
          <Slider label="Drums CFG" value={params.cfg_drums} min={-1} max={7} step={0.1} onChange={cfg_drums => setParams({...params, cfg_drums})} />
          <Slider label="Unmask width" value={params.unmask_width} min={0} max={127} step={1} onChange={unmask_width => setParams({...params, unmask_width})} />
        </section>
        <section><div className="section-heading"><div><span>CONTEXT</span><h2>State banks</h2></div><RefreshCw /></div>
          {[0, 1, 2].map(index => <div className="bank" key={index}><strong>Bank {index + 1}</strong>
            <button onClick={() => send({type: 'bank', action: 'save', index})}>Save</button>
            <button onClick={() => send({type: 'bank', action: 'load', index})}>Recall</button></div>)}
          <p className="fine-print">Banks live in GPU memory and reset when the model changes.</p>
        </section>
        <section className="diagnostics"><div className="section-heading"><div><span>CUDA</span><h2>Runtime</h2></div></div>
          <Metric label="Model" value={status.model.replace('mrt2_', '')} />
          <Metric label="Audio rate" value="48 kHz stereo" />
          <Metric label="Control frame" value="40 ms" />
        </section>
      </div>}
    </main>
  </div>;
}

declare global {
  interface Navigator {requestMIDIAccess(): Promise<MIDIAccess>}
}
