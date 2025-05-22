
import React, { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MessageCircle, Send, X, Mic, MicOff } from "lucide-react";
import { storeResponse } from '@/utils/storage';
import { useToast } from "@/hooks/use-toast";
import { accidentQuestions } from '@/data/accidentQuestions';
import { supabase } from '@/integrations/supabase/client';
import { v4 as uuidv4 } from 'uuid';

type Message = {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  isQuestion?: boolean;
  questionId?: number;
};

const initialMessages: Message[] = [
  {
    id: 1,
    text: "Hello! I'm here to help you report your accident. Let me ask you some questions to gather all the necessary details.",
    sender: 'bot'
  }
];

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [sessionId] = useState<string>(() => uuidv4());
  
  // Reference for speech recognition
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    // Check if browser supports speech recognition
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');
        
        setTranscript(transcript);
        setInput(transcript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        setIsListening(false);
        toast({
          title: "Speech Recognition Error",
          description: "There was a problem with speech recognition. Please try again.",
        });
      };
    }
  }, [toast]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Ask the first question after initial greeting
    if (messages.length === 1) {
      setTimeout(() => {
        askNextQuestion();
      }, 1000);
    }
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleListening = () => {
    if (!recognitionRef.current) {
      toast({
        title: "Speech Recognition Not Supported",
        description: "Your browser doesn't support speech recognition. Please type your answer.",
      });
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setInput('');
      setTranscript('');
      recognitionRef.current.start();
      setIsListening(true);
      toast({
        title: "Listening...",
        description: "Please speak clearly. Click the mic button again to stop.",
      });
    }
  };

  const askNextQuestion = () => {
    if (currentQuestionIndex < accidentQuestions.length) {
      const question = accidentQuestions[currentQuestionIndex];
      const newMessage: Message = {
        id: messages.length + 1,
        text: question.text,
        sender: 'bot',
        isQuestion: true,
        questionId: question.id
      };
      
      setMessages(prev => [...prev, newMessage]);
      setCurrentQuestionIndex(prevIndex => prevIndex + 1);
    } else {
      // All questions have been asked
      const finalMessage: Message = {
        id: messages.length + 1,
        text: "Thank you for providing all the information. Your accident report has been submitted successfully. Your case number is #AC-2023-7845.",
        sender: 'bot'
      };
      
      setMessages(prev => [...prev, finalMessage]);
      
      toast({
        title: "Report Submitted",
        description: "Your accident report has been successfully recorded.",
      });
    }
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;
    
    // Add user message
    const newMessage: Message = {
      id: messages.length + 1,
      text: input,
      sender: 'user'
    };
    
    setMessages([...messages, newMessage]);
    
    // Find the last question that was asked
    const lastQuestion = [...messages].reverse().find(msg => msg.isQuestion);
    
    // Store the response locally
    if (lastQuestion?.questionId) {
      storeResponse('chat', { 
        question: lastQuestion.text, 
        answer: input 
      });
      
      // Store in Supabase
      try {
        await supabase.from('user_responses').insert({
          session_id: sessionId,
          question_id: lastQuestion.questionId,
          question: lastQuestion.text,
          answer: input,
          category: accidentQuestions.find(q => q.id === lastQuestion.questionId)?.category || null,
          response_type: 'chat'
        });
      } catch (error) {
        console.error('Error storing response in Supabase:', error);
      }
    }
    
    setInput('');
    if (isListening) {
      toggleListening();
    }
    
    // Wait a bit and then ask the next question
    setTimeout(() => {
      askNextQuestion();
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <>
      {/* Chat toggle button */}
      {!isOpen && (
        <button 
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 bg-brand-600 text-white p-4 rounded-full shadow-lg hover:bg-brand-700 transition-colors z-50"
        >
          <MessageCircle className="h-6 w-6" />
        </button>
      )}
      
      {/* Chat window */}
      {isOpen && (
        <div className="fixed bottom-6 right-6 w-80 sm:w-96 h-[500px] bg-white rounded-lg shadow-xl flex flex-col z-50 border border-gray-200">
          <div className="bg-brand-600 text-white px-4 py-3 rounded-t-lg flex justify-between items-center">
            <div className="flex items-center">
              <span className="h-2 w-2 rounded-full bg-green-400 mr-2"></span>
              <h3 className="font-semibold">Accident Assistant</h3>
            </div>
            <button onClick={() => setIsOpen(false)}>
              <X className="h-5 w-5" />
            </button>
          </div>
          
          <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
            <div className="space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`${
                    msg.sender === 'user' 
                      ? 'bg-gray-100 ml-auto rounded-tl-lg rounded-bl-lg rounded-tr-none' 
                      : 'bg-brand-50 rounded-tr-lg rounded-br-lg rounded-tl-none'
                  } p-3 rounded-lg max-w-[75%]`}
                >
                  <p className="text-sm">{msg.text}</p>
                  {msg.isQuestion && (
                    <p className="text-xs text-gray-500 mt-1">Question {currentQuestionIndex} of {accidentQuestions.length}</p>
                  )}
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          </div>
          
          <div className="p-3 border-t border-gray-200">
            <div className="flex items-center gap-2">
              <Input
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                className="flex-1"
              />
              <Button
                size="icon"
                variant="outline"
                onClick={toggleListening}
                className={`shrink-0 ${isListening ? 'bg-red-100' : ''}`}
                title={isListening ? "Stop listening" : "Start voice input"}
              >
                {isListening ? (
                  <MicOff className="h-4 w-4" />
                ) : (
                  <Mic className="h-4 w-4" />
                )}
              </Button>
              <Button size="icon" onClick={handleSendMessage} className="shrink-0">
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ChatWidget;
