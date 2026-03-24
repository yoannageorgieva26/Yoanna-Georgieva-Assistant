/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Modality } from "@google/genai";
import { motion, AnimatePresence } from "motion/react";
import { Mic, MicOff, Volume2, VolumeX, Info, Sparkles, X } from "lucide-react";

// System Instruction based on the provided resume
const SYSTEM_INSTRUCTION = `
You are "Yoanna Online", the official voice assistant for Yoanna Georgieva. 
Yoanna is a Business of Fashion student at ESMOD Paris (2024-2027).
You are chic, sophisticated, professional, and helpful. 
You speak English and French fluently with a regular American accent.

Your knowledge is based on Yoanna's resume:
- Education: ESMOD Paris (Business of Fashion), Illinois Mathematics and Science Academy (IMSA).
- Experience: 
  - Fashion Republic Magazine: Assistant Directeur Défilé (Fashion Week AH 26/27).
  - Agora Paris (The Hideout Clothing): Sales Intern (June 2025 - July 2025).
  - Vivienne Westwood: Show Dresser / Assistant Défilé (AW 25/26 Fashion Week).
  - Pilot Company: Sales Associate.
  - Global Textile Trading: Office Assistant.
- Skills: Data Analysis, Logicality, Adaptability, Leadership, Microsoft Office, Google Suite, Adobe Suite, Social Media.
- Languages: English (Native), Bulgarian (Native), French (B1).
- Certifications: Marist Summer Pre-College (Fashion Merchandising), Parsons Paris Fashion & Luxury Online Courses (Inside the Business of Fashion and Luxury).

Tone: Elegant, concise, and high-fashion. 
Initial Greeting: "Hello, I'm Yoanna Online. What would you like to know about Yoanna?"
If asked about her accent, confirm it's a regular American accent.
If asked to speak French, do so elegantly.
`;

export default function App() {
  const [isActive, setIsActive] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sessionRef = useRef<any>(null);
  const audioQueueRef = useRef<Int16Array[]>([]);
  const isPlayingRef = useRef(false);

  const stopAssistant = useCallback(() => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsActive(false);
    setIsListening(false);
    setIsSpeaking(false);
  }, []);

  const playNextInQueue = useCallback(async () => {
    if (audioQueueRef.current.length === 0 || isPlayingRef.current || !audioContextRef.current) {
      isPlayingRef.current = false;
      setIsSpeaking(false);
      return;
    }

    isPlayingRef.current = true;
    setIsSpeaking(true);
    const chunk = audioQueueRef.current.shift()!;
    
    const audioBuffer = audioContextRef.current.createBuffer(1, chunk.length, 16000);
    const channelData = audioBuffer.getChannelData(0);
    for (let i = 0; i < chunk.length; i++) {
      channelData[i] = chunk[i] / 32768.0;
    }

    const source = audioContextRef.current.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContextRef.current.destination);
    source.onended = () => {
      isPlayingRef.current = false;
      playNextInQueue();
    };
    source.start();
  }, []);

  const startAssistant = async () => {
    try {
      setError(null);
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const session = await ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-12-2025",
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Puck" } }, // Puck has a nice professional tone
          },
          systemInstruction: SYSTEM_INSTRUCTION,
        },
        callbacks: {
          onopen: () => {
            setIsActive(true);
            setIsListening(true);
            
            // Start processing microphone input
            const source = audioContextRef.current!.createMediaStreamSource(stream);
            const processor = audioContextRef.current!.createScriptProcessor(4096, 1, 1);
            processorRef.current = processor;

            processor.onaudioprocess = (e) => {
              if (!sessionRef.current) return;
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmData = new Int16Array(inputData.length);
              for (let i = 0; i < inputData.length; i++) {
                pcmData[i] = Math.max(-1, Math.min(1, inputData[i])) * 32767;
              }
              const base64Data = btoa(String.fromCharCode(...new Uint8Array(pcmData.buffer)));
              sessionRef.current.sendRealtimeInput({
                audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
              });
            };

            source.connect(processor);
            processor.connect(audioContextRef.current!.destination);
          },
          onmessage: async (message) => {
            if (message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data) {
              const base64Audio = message.serverContent.modelTurn.parts[0].inlineData.data;
              const binaryString = atob(base64Audio);
              const bytes = new Uint8Array(binaryString.length);
              for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
              }
              const pcmData = new Int16Array(bytes.buffer);
              audioQueueRef.current.push(pcmData);
              if (!isPlayingRef.current) {
                playNextInQueue();
              }
            }
            if (message.serverContent?.interrupted) {
              audioQueueRef.current = [];
              isPlayingRef.current = false;
              setIsSpeaking(false);
            }
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setError("Connection lost. Please try again.");
            stopAssistant();
          },
          onclose: () => {
            stopAssistant();
          }
        }
      });

      sessionRef.current = session;
    } catch (err) {
      console.error("Failed to start assistant:", err);
      setError("Could not access microphone or connect to AI.");
    }
  };

  useEffect(() => {
    return () => stopAssistant();
  }, [stopAssistant]);

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 relative overflow-hidden chic-gradient">
      {/* Background Decorative Elements */}
      <div className="absolute top-0 left-0 w-full h-full pointer-events-none opacity-20">
        <div className="absolute top-10 left-10 w-64 h-64 border border-luxury-gold/20 rounded-full blur-3xl" />
        <div className="absolute bottom-10 right-10 w-96 h-96 border border-luxury-gold/10 rounded-full blur-3xl" />
      </div>

      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-12 text-center z-10"
      >
        <h1 className="font-serif text-5xl md:text-7xl tracking-tighter uppercase font-light mb-2">
          Yoanna <span className="italic font-extralight text-luxury-gold/80">Online</span>
        </h1>
        <p className="text-xs uppercase tracking-[0.4em] text-luxury-black/50 font-medium">
          Official Voice Assistant
        </p>
      </motion.header>

      {/* Main Interaction Area */}
      <main className="flex flex-col items-center justify-center z-10 w-full max-w-md">
        <div className="relative mb-12">
          {/* Pulse Rings */}
          <AnimatePresence>
            {(isActive || isSpeaking) && (
              <>
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1.5, opacity: 0.1 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  transition={{ repeat: Infinity, duration: 2, ease: "easeOut" }}
                  className="absolute inset-0 rounded-full border-2 border-luxury-gold"
                />
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 2, opacity: 0.05 }}
                  exit={{ scale: 0.8, opacity: 0 }}
                  transition={{ repeat: Infinity, duration: 3, ease: "easeOut", delay: 0.5 }}
                  className="absolute inset-0 rounded-full border border-luxury-gold"
                />
              </>
            )}
          </AnimatePresence>

          {/* Main Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={isActive ? stopAssistant : startAssistant}
            className={`relative w-48 h-48 rounded-full flex items-center justify-center transition-all duration-700 shadow-2xl ${
              isActive 
                ? 'bg-luxury-black text-luxury-paper' 
                : 'bg-white border border-luxury-gold/20 text-luxury-black'
            }`}
          >
            <AnimatePresence mode="wait">
              {isActive ? (
                <motion.div
                  key="active"
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.5 }}
                  className="flex flex-col items-center"
                >
                  {isSpeaking ? (
                    <Volume2 className="w-12 h-12 mb-2 text-luxury-gold animate-pulse" />
                  ) : (
                    <Mic className="w-12 h-12 mb-2 text-luxury-gold" />
                  )}
                  <span className="text-[10px] uppercase tracking-widest font-bold">
                    {isSpeaking ? "Speaking" : "Listening"}
                  </span>
                </motion.div>
              ) : (
                <motion.div
                  key="inactive"
                  initial={{ opacity: 0, scale: 0.5 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.5 }}
                  className="flex flex-col items-center"
                >
                  <Sparkles className="w-12 h-12 mb-2 text-luxury-gold/60" />
                  <span className="text-[10px] uppercase tracking-widest font-bold">
                    Begin Experience
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>
        </div>

        {/* Status Text */}
        <div className="h-12 flex items-center justify-center text-center px-4">
          <AnimatePresence mode="wait">
            {error ? (
              <motion.p 
                key="error"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-red-500 text-sm font-medium"
              >
                {error}
              </motion.p>
            ) : isActive ? (
              <motion.p 
                key="status"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-luxury-black/60 text-sm font-serif italic"
              >
                {isSpeaking ? "Yoanna is sharing her story..." : "Ask about her fashion journey"}
              </motion.p>
            ) : (
              <motion.p 
                key="welcome"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="text-luxury-black/40 text-sm max-w-xs"
              >
                Experience the professional world of Yoanna Georgieva through her personalized AI assistant.
              </motion.p>
            )}
          </AnimatePresence>
        </div>
      </main>

      {/* Footer Controls */}
      <footer className="absolute bottom-12 flex gap-8 z-10">
        <button 
          onClick={() => setShowInfo(true)}
          className="p-3 rounded-full border border-luxury-gold/20 hover:bg-white transition-colors"
        >
          <Info className="w-5 h-5 text-luxury-gold" />
        </button>
        {isActive && (
          <button 
            onClick={stopAssistant}
            className="p-3 rounded-full border border-red-200 hover:bg-red-50 transition-colors"
          >
            <X className="w-5 h-5 text-red-400" />
          </button>
        )}
      </footer>

      {/* Info Modal */}
      <AnimatePresence>
        {showInfo && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-luxury-black/90 backdrop-blur-sm z-50 flex items-center justify-center p-6"
          >
            <motion.div 
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="bg-luxury-paper max-w-lg w-full p-8 rounded-3xl relative overflow-hidden"
            >
              <button 
                onClick={() => setShowInfo(false)}
                className="absolute top-6 right-6 p-2 hover:bg-luxury-black/5 rounded-full transition-colors"
              >
                <X className="w-6 h-6" />
              </button>

              <h2 className="font-serif text-3xl mb-6">About Yoanna</h2>
              
              <div className="space-y-4 text-sm text-luxury-black/70 leading-relaxed">
                <p>
                  Yoanna Georgieva is a visionary <span className="font-bold">Business of Fashion</span> student at ESMOD Paris. 
                  With a background spanning from IMSA to the heart of the Parisian fashion scene, she combines analytical precision with creative flair.
                </p>
                <div className="grid grid-cols-2 gap-4 pt-4 border-t border-luxury-gold/10">
                  <div>
                    <h3 className="text-[10px] uppercase tracking-widest font-bold text-luxury-gold mb-2">Education</h3>
                    <p className="font-medium">ESMOD Paris</p>
                    <p className="text-xs">Business of Fashion</p>
                  </div>
                  <div>
                    <h3 className="text-[10px] uppercase tracking-widest font-bold text-luxury-gold mb-2">Experience</h3>
                    <p className="font-medium">Vivienne Westwood</p>
                    <p className="text-xs">Show Dresser</p>
                  </div>
                </div>
                <p className="pt-4 text-xs italic">
                  "Yoanna Online" is a bespoke AI experience designed to showcase her professional journey in an immersive, voice-first format.
                </p>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
