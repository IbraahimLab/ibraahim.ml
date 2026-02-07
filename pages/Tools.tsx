import React, { useEffect, useMemo, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle2, Loader2, Mic, Square, Upload, Send, Link as LinkIcon } from 'lucide-react';

const MAX_DURATION_SECONDS = 30;
const TARGET_SAMPLE_RATE = 16000;
const ENV = (import.meta as ImportMeta & { env?: Record<string, string> }).env ?? {};
const RAW_API_BASE_URL = ENV.VITE_API_BASE_URL?.trim() ?? '';
const USE_ABSOLUTE_API_IN_DEV = ENV.VITE_USE_ABSOLUTE_API_IN_DEV === 'true';
const API_BASE_URL = (ENV.DEV && !USE_ABSOLUTE_API_IN_DEV) ? '' : RAW_API_BASE_URL.replace(/\/+$/, '');

type AudioSource = 'record' | 'upload';

interface SubmitResponse {
  ok: boolean;
  dataset_repo: string;
  sample_id: string;
  audio_path: string;
  jsonl_path: string;
  duration_sec: number;
  commit_url?: string;
}

type WindowWithWebkitAudio = Window & typeof globalThis & {
  webkitAudioContext?: new () => AudioContext;
};

const writeString = (view: DataView, offset: number, value: string): void => {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
};

const encodeMonoAudioBufferToWav = (audioBuffer: AudioBuffer): Blob => {
  const samples = audioBuffer.getChannelData(0);
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = audioBuffer.sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, audioBuffer.sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const intSample = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(offset, intSample, true);
    offset += bytesPerSample;
  }

  return new Blob([buffer], { type: 'audio/wav' });
};

const normalizeToMono16kWav = async (blob: Blob): Promise<{ wavBlob: Blob; durationSec: number }> => {
  const maybeCtor = (window.AudioContext || (window as WindowWithWebkitAudio).webkitAudioContext);
  if (!maybeCtor) {
    throw new Error('Audio APIs are not supported in this browser.');
  }

  const audioContext = new maybeCtor();
  try {
    const sourceBuffer = await blob.arrayBuffer();
    const decodedBuffer = await audioContext.decodeAudioData(sourceBuffer.slice(0));
    const frameCount = Math.ceil(decodedBuffer.duration * TARGET_SAMPLE_RATE);
    const offlineContext = new OfflineAudioContext(1, frameCount, TARGET_SAMPLE_RATE);
    const monoBuffer = offlineContext.createBuffer(1, decodedBuffer.length, decodedBuffer.sampleRate);
    const monoData = monoBuffer.getChannelData(0);

    for (let channel = 0; channel < decodedBuffer.numberOfChannels; channel += 1) {
      const input = decodedBuffer.getChannelData(channel);
      for (let i = 0; i < input.length; i += 1) {
        monoData[i] += input[i] / decodedBuffer.numberOfChannels;
      }
    }

    const sourceNode = offlineContext.createBufferSource();
    sourceNode.buffer = monoBuffer;
    sourceNode.connect(offlineContext.destination);
    sourceNode.start(0);

    const rendered = await offlineContext.startRendering();
    const durationSec = Number(rendered.duration.toFixed(2));
    return { wavBlob: encodeMonoAudioBufferToWav(rendered), durationSec };
  } finally {
    await audioContext.close();
  }
};

export const Tools: React.FC = () => {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const recordingTimeoutRef = useRef<number | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const previewUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [isRecording, setIsRecording] = useState(false);
  const [isPreparingAudio, setIsPreparingAudio] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string>('');
  const [audioDuration, setAudioDuration] = useState<number | null>(null);
  const [audioSource, setAudioSource] = useState<AudioSource | null>(null);
  const [transcript, setTranscript] = useState('');
  const [language, setLanguage] = useState('en');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [submitResult, setSubmitResult] = useState<SubmitResponse | null>(null);

  const canSubmit = useMemo(() => {
    return Boolean(audioBlob && transcript.trim() && !isSubmitting && !isPreparingAudio && (audioDuration ?? 0) <= MAX_DURATION_SECONDS);
  }, [audioBlob, transcript, isSubmitting, isPreparingAudio, audioDuration]);

  const clearRecordingTimeout = (): void => {
    if (recordingTimeoutRef.current !== null) {
      clearTimeout(recordingTimeoutRef.current);
      recordingTimeoutRef.current = null;
    }
  };

  const stopMediaStream = (): void => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  };

  const replacePreviewUrl = (nextUrl: string): void => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
    previewUrlRef.current = nextUrl;
    setAudioPreviewUrl(nextUrl);
  };

  const clearAudioSelection = (): void => {
    setAudioBlob(null);
    setAudioDuration(null);
    setAudioSource(null);
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setAudioPreviewUrl('');
  };

  const prepareAudio = async (inputBlob: Blob, source: AudioSource): Promise<void> => {
    setIsPreparingAudio(true);
    setErrorMessage('');
    setSubmitResult(null);
    try {
      const { wavBlob, durationSec } = await normalizeToMono16kWav(inputBlob);
      if (durationSec > MAX_DURATION_SECONDS) {
        clearAudioSelection();
        setErrorMessage(`Audio must be ${MAX_DURATION_SECONDS} seconds or less. Current duration: ${durationSec}s.`);
        return;
      }

      setAudioBlob(wavBlob);
      setAudioDuration(durationSec);
      setAudioSource(source);
      replacePreviewUrl(URL.createObjectURL(wavBlob));
    } catch (error) {
      clearAudioSelection();
      setErrorMessage(error instanceof Error ? error.message : 'Unable to process audio. Try a different file format.');
    } finally {
      setIsPreparingAudio(false);
    }
  };

  const startRecording = async (): Promise<void> => {
    if (isRecording || isPreparingAudio) {
      return;
    }

    setErrorMessage('');
    setSubmitResult(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMessage('Microphone recording is not supported in this browser.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      chunksRef.current = [];

      const preferredType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : undefined;
      const recorder = preferredType ? new MediaRecorder(stream, { mimeType: preferredType }) : new MediaRecorder(stream);

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        clearRecordingTimeout();
        stopMediaStream();
        setIsRecording(false);
        if (chunksRef.current.length === 0) {
          setErrorMessage('No audio was captured. Please try again.');
          return;
        }
        const recordedBlob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        await prepareAudio(recordedBlob, 'record');
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      recordingTimeoutRef.current = window.setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          mediaRecorderRef.current.stop();
        }
      }, MAX_DURATION_SECONDS * 1000);
    } catch {
      setIsRecording(false);
      stopMediaStream();
      setErrorMessage('Microphone access was blocked or unavailable.');
    }
  };

  const stopRecording = (): void => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
      return;
    }
    mediaRecorderRef.current.stop();
  };

  const onPickFile = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await prepareAudio(file, 'upload');
    event.target.value = '';
  };

  const submitSample = async (): Promise<void> => {
    if (!audioBlob) {
      setErrorMessage('Please record or upload an audio sample first.');
      return;
    }
    if (!transcript.trim()) {
      setErrorMessage('Transcript is required.');
      return;
    }

    setIsSubmitting(true);
    setErrorMessage('');
    setSubmitResult(null);

    try {
      const form = new FormData();
      const generatedId = typeof crypto.randomUUID === 'function' ? crypto.randomUUID() : `${Date.now()}`;
      form.append('audio', audioBlob, `${generatedId}.wav`);
      form.append('transcript', transcript.trim());
      form.append('language', language.trim() || 'en');

      const response = await fetch(`${API_BASE_URL}/api/tools/voice-dataset/submit`, {
        method: 'POST',
        body: form,
      });

      let data: unknown = null;
      try {
        data = await response.json();
      } catch {
        data = null;
      }
      if (!response.ok) {
        const maybeDetail = typeof data === 'object' && data !== null && 'detail' in data
          ? (data as { detail?: unknown }).detail
          : null;
        throw new Error(typeof maybeDetail === 'string' ? maybeDetail : `Upload failed with status ${response.status}.`);
      }

      setSubmitResult(data as SubmitResponse);
    } catch (error) {
      if (error instanceof TypeError) {
        setErrorMessage(
          `Cannot reach API (${API_BASE_URL || 'same-origin /api'}). Check backend URL, HTTPS/HTTP mismatch, and ALLOWED_ORIGINS.`
        );
      } else {
        setErrorMessage(error instanceof Error ? error.message : 'Upload failed.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    return () => {
      clearRecordingTimeout();
      stopMediaStream();
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      className="pt-48 max-w-5xl mx-auto px-4 pb-32"
    >
      <div className="max-w-3xl mb-14">
        <h1 className="text-5xl font-black mb-5 text-slate-900 dark:text-white">Tools</h1>
        <p className="text-lg text-slate-500 dark:text-slate-400 font-medium">
          Tool #1: Voice Dataset Uploader. Record or upload audio, add a manual transcript, then submit directly to your public Hugging Face dataset.
        </p>
      </div>

      <section className="p-8 md:p-10 bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] rounded-[2rem]">
        <div className="flex items-center justify-between gap-4 flex-wrap mb-8">
          <h2 className="text-2xl font-black text-slate-900 dark:text-white">Voice Dataset Uploader</h2>
          <span className="text-xs font-black uppercase tracking-widest text-blue-600">Limit: 30 seconds</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-7">
          <button
            type="button"
            onClick={isRecording ? stopRecording : startRecording}
            disabled={isPreparingAudio}
            className="px-5 py-4 rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/[0.02] hover:border-blue-500 transition-all flex items-center justify-center gap-2 font-semibold disabled:opacity-60"
          >
            {isRecording ? <Square size={16} /> : <Mic size={16} />}
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </button>

          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isPreparingAudio || isRecording}
            className="px-5 py-4 rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/[0.02] hover:border-blue-500 transition-all flex items-center justify-center gap-2 font-semibold disabled:opacity-60"
          >
            <Upload size={16} />
            Upload Audio File
          </button>
          <input ref={fileInputRef} type="file" accept="audio/*" className="hidden" onChange={onPickFile} />
        </div>

        {isPreparingAudio && (
          <div className="mb-7 text-sm font-semibold text-slate-500 dark:text-slate-300 flex items-center gap-2">
            <Loader2 size={16} className="animate-spin" />
            Normalizing audio to 16kHz mono WAV...
          </div>
        )}

        {audioPreviewUrl && (
          <div className="mb-8 p-5 rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-50/60 dark:bg-white/[0.02]">
            <audio controls src={audioPreviewUrl} className="w-full mb-3" />
            <div className="text-sm text-slate-500 dark:text-slate-400">
              <span className="font-semibold text-slate-700 dark:text-slate-300">Source:</span> {audioSource}
              {'  '}|{'  '}
              <span className="font-semibold text-slate-700 dark:text-slate-300">Duration:</span> {audioDuration ?? '--'}s
            </div>
          </div>
        )}

        <div className="space-y-5">
          <div>
            <label className="block text-sm font-semibold mb-2 text-slate-700 dark:text-slate-300">Manual Transcript *</label>
            <textarea
              value={transcript}
              onChange={(event) => setTranscript(event.target.value)}
              placeholder="Type exactly what is spoken in the audio sample..."
              rows={4}
              className="w-full px-4 py-3 rounded-2xl bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] focus:ring-2 focus:ring-blue-600 outline-none transition-all text-slate-900 dark:text-white"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold mb-2 text-slate-700 dark:text-slate-300">Language</label>
              <input
                type="text"
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                placeholder="en"
                className="w-full px-4 py-3 rounded-2xl bg-white dark:bg-white/[0.03] border border-slate-200 dark:border-white/[0.08] focus:ring-2 focus:ring-blue-600 outline-none transition-all text-slate-900 dark:text-white"
              />
            </div>
          </div>
        </div>

        <div className="mt-8 flex items-center gap-3 flex-wrap">
          <button
            type="button"
            onClick={submitSample}
            disabled={!canSubmit}
            className="px-6 py-3 rounded-full bg-blue-600 text-white font-bold hover:scale-105 transition-transform disabled:opacity-50 disabled:hover:scale-100 flex items-center gap-2"
          >
            {isSubmitting ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            {isSubmitting ? 'Submitting...' : 'Submit to Hugging Face'}
          </button>
          {!canSubmit && (
            <span className="text-sm text-slate-500 dark:text-slate-400">Provide audio and transcript to enable submission.</span>
          )}
        </div>

        {errorMessage && (
          <div className="mt-6 p-4 rounded-2xl border border-rose-200 bg-rose-50 text-rose-700 text-sm font-medium flex items-start gap-2">
            <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
            <span>{errorMessage}</span>
          </div>
        )}

        {submitResult && (
          <div className="mt-6 p-4 rounded-2xl border border-emerald-200 bg-emerald-50 text-emerald-800 text-sm">
            <div className="font-semibold flex items-center gap-2 mb-2">
              <CheckCircle2 size={16} />
              Sample uploaded successfully
            </div>
            <div>Dataset: {submitResult.dataset_repo}</div>
            <div>Sample ID: {submitResult.sample_id}</div>
            <div>Audio Path: {submitResult.audio_path}</div>
            <div>JSONL Path: {submitResult.jsonl_path}</div>
            {submitResult.commit_url && (
              <a
                href={submitResult.commit_url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-blue-700 underline mt-2"
              >
                <LinkIcon size={14} />
                View commit
              </a>
            )}
          </div>
        )}
      </section>
    </motion.div>
  );
};
