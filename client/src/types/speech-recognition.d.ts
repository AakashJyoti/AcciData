
interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
  isFinal: boolean;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  maxAlternatives: number;
  onaudioend: (ev: Event) => any;
  onaudiostart: (ev: Event) => any;
  onend: (ev: Event) => any;
  onerror: (ev: SpeechRecognitionErrorEvent) => any;
  onnomatch: (ev: SpeechRecognitionEvent) => any;
  onresult: (ev: SpeechRecognitionEvent) => any;
  onsoundend: (ev: Event) => any;
  onsoundstart: (ev: Event) => any;
  onspeechend: (ev: Event) => any;
  onspeechstart: (ev: Event) => any;
  onstart: (ev: Event) => any;
  abort(): void;
  start(): void;
  stop(): void;
}

interface Window {
  SpeechRecognition: new () => SpeechRecognition;
  webkitSpeechRecognition: new () => SpeechRecognition;
}
