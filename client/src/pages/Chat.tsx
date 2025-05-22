import React, { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Mic, MicOff, Upload, Paperclip } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { accidentQuestions } from "@/data/accidentQuestions";
import { supabase } from "@/integrations/supabase/client";
import { v4 as uuidv4 } from "uuid";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FileUpload from "@/components/FileUpload";
import FileDisplay from "@/components/FileDisplay";
import axios from "axios";
import { APIUrl } from "@/constant";

type Message = {
  id: number;
  text: string;
  sender: "user" | "bot";
  isQuestion?: boolean;
  questionId?: number;
  fileUrl?: string;
  fileName?: string;
  fileType?: string;
  isFile?: boolean;
};

// Generate a unique ID for initial message
const uniqueId = Date.now() + Math.floor(Math.random() * 1000);

const initialMessages: Message[] = [
  {
    id: uniqueId,
    text: "Hello! I'm here to help you report your accident. Let me ask you some questions to gather all the necessary details.",
    sender: "bot",
  },
];

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [sessionId, setSessionId] = useState<string>("");
  const [showFileUpload, setShowFileUpload] = useState(false);

  // Reference for speech recognition
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    // Check if browser supports speech recognition
    if ("SpeechRecognition" in window || "webkitSpeechRecognition" in window) {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      recognitionRef.current.onresult = (event) => {
        const transcript = Array.from(event.results)
          .map((result) => result[0])
          .map((result) => result.transcript)
          .join("");

        setTranscript(transcript);
        setInput(transcript);
      };

      recognitionRef.current.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
        toast({
          title: "Speech Recognition Error",
          description:
            "There was a problem with speech recognition. Please try again.",
        });
      };
    }
  }, [toast]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleListening = () => {
    if (!recognitionRef.current) {
      toast({
        title: "Speech Recognition Not Supported",
        description:
          "Your browser doesn't support speech recognition. Please type your answer.",
      });
      return;
    }

    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      setInput("");
      setTranscript("");
      recognitionRef.current.start();
      setIsListening(true);
      toast({
        title: "Listening...",
        description:
          "Please speak clearly. Click the mic button again to stop.",
      });
    }
  };

  // Generate a unique ID for messages
  const generateUniqueId = () => {
    return Date.now() + Math.floor(Math.random() * 1000);
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    // Add user message
    const newMessage: Message = {
      id: generateUniqueId(),
      text: input,
      sender: "user",
    };

    setMessages((prev) => [...prev, newMessage]);

    try {
      const { data } = await axios.post(`${APIUrl}/chat`, {
        user_input: input,
        session_id: sessionId,
      });

      setInput("");

      if (data && data.response) {
        const newMessage: Message = {
          id: generateUniqueId(),
          text: data.response,
          sender: "bot",
        };

        setMessages((prev) => [...prev, newMessage]);
      } else {
        console.error("Failed to get session ID");
      }
    } catch (error) {
      console.error("Error storing response in Supabase:", error);
    }

    if (isListening) {
      toggleListening();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSendMessage();
    }
  };

  // Toggle file upload UI
  const toggleFileUpload = () => {
    setShowFileUpload((prev) => !prev);
  };

  // Handle file upload completion
  const handleFileUploadComplete = (
    fileUrl: string,
    fileName: string,
    fileType: string
  ) => {
    // Add a file message
    const fileMessage: Message = {
      id: generateUniqueId(),
      text: `[File: ${fileName}]`,
      sender: "user",
      fileUrl,
      fileName,
      fileType,
      isFile: true,
    };

    setMessages([...messages, fileMessage]);

    // Find the last question that was asked
    const lastQuestion = [...messages].reverse().find((msg) => msg.isQuestion);

    // Store in Supabase
    if (lastQuestion?.questionId) {
      try {
        // Store file metadata in the documents table
        supabase.from("documents").insert({
          session_id: sessionId,
          filename: fileName,
          file_url: fileUrl,
          file_type: fileType,
        });

        // Store a reference to the file in user_responses
        supabase.from("user_responses").insert({
          session_id: sessionId,
          question_id: lastQuestion.questionId,
          question: lastQuestion.text,
          answer: `[File uploaded: ${fileName}]`,
          category:
            accidentQuestions.find((q) => q.id === lastQuestion.questionId)
              ?.category || null,
          response_type: "chat",
        });
      } catch (error) {
        console.error("Error storing file data in Supabase:", error);
      }
    }

    // Hide the file upload UI
    setShowFileUpload(false);
  };

  useEffect(() => {
    const getSessionId = async () => {
      const { data } = await axios.post(`${APIUrl}/new_session`, {
        user_id: uuidv4(),
      });
      if (data && data.session_id) {
        setSessionId(data.session_id);
      } else {
        console.error("Failed to get session ID");
      }
    };
    if (sessionId === "") {
      getSessionId();
    }
  }, [sessionId]);

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1">
        <div className="container mx-auto px-4 py-12">
          <div className="max-w-3xl mx-auto">
            <h1 className="text-3xl font-bold mb-8 text-center text-gray-900">
              Chat Accident Assistant
            </h1>

            <div className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden">
              <div className="bg-brand-600 text-white px-4 py-3 flex items-center">
                <span className="h-2 w-2 rounded-full bg-green-400 mr-2"></span>
                <h3 className="font-semibold">Accident Assistant</h3>
              </div>

              <div className="h-[500px] overflow-y-auto p-4 bg-gray-50">
                <div className="space-y-4">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`${
                        msg.sender === "user"
                          ? "bg-gray-100 ml-auto rounded-tl-lg rounded-bl-lg rounded-tr-none"
                          : "bg-brand-50 rounded-tr-lg rounded-br-lg rounded-tl-none"
                      } p-3 rounded-lg max-w-[75%]`}
                    >
                      <p className="text-sm">{msg.text}</p>
                      {msg.isQuestion && (
                        <p className="text-xs text-gray-500 mt-1">
                          Question {currentQuestionIndex} of{" "}
                          {accidentQuestions.length}
                        </p>
                      )}
                      {msg.isFile &&
                        msg.fileUrl &&
                        msg.fileName &&
                        msg.fileType && (
                          <div className="mt-2">
                            <FileDisplay
                              fileUrl={msg.fileUrl}
                              fileName={msg.fileName}
                              fileType={msg.fileType}
                              showDownload={true}
                            />
                          </div>
                        )}
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>
              </div>

              <div className="p-3 border-t border-gray-200">
                {showFileUpload ? (
                  <div className="mb-2">
                    <FileUpload
                      sessionId={sessionId}
                      onUploadComplete={handleFileUploadComplete}
                    />
                    <div className="mt-2 flex justify-end">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={toggleFileUpload}
                      >
                        Cancel
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <Input
                      placeholder="Type your message..."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyPress={handleKeyPress}
                      className="flex-1"
                    />
                    {/* <Button
                      size="icon"
                      variant="outline"
                      onClick={toggleFileUpload}
                      className="shrink-0"
                      title="Upload a file"
                    >
                      <Paperclip className="h-4 w-4" />
                    </Button> */}
                    {/* <Button
                      size="icon"
                      variant="outline"
                      onClick={toggleListening}
                      className={`shrink-0 ${isListening ? "bg-red-100" : ""}`}
                      title={
                        isListening ? "Stop listening" : "Start voice input"
                      }
                    >
                      {isListening ? (
                        <MicOff className="h-4 w-4" />
                      ) : (
                        <Mic className="h-4 w-4" />
                      )}
                    </Button> */}
                    <Button
                      size="icon"
                      onClick={handleSendMessage}
                      className="shrink-0"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Chat;
