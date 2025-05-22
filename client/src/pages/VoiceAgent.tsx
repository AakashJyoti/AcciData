
import React, { useState, useEffect, useRef } from 'react';
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Mic, MicOff, Volume2, AlertCircle, ChevronLeft, ChevronRight, Zap, ZapOff, Upload, PaperclipIcon } from "lucide-react";
import { storeResponse } from '@/utils/storage';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import { useToast } from "@/hooks/use-toast";
import { accidentQuestions, Question, getQuestionsByCategory } from '@/data/accidentQuestions';
import { supabase } from '@/integrations/supabase/client';
import { Switch } from "@/components/ui/switch";
import FileUpload from '@/components/FileUpload';
import FileDisplay from '@/components/FileDisplay';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { v4 as uuidv4 } from 'uuid';

const VoiceAgent: React.FC = () => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [responses, setResponses] = useState<Record<number, string>>({});
  const [isComplete, setIsComplete] = useState(false);
  const { toast } = useToast();
  const [activeCategory, setActiveCategory] = useState<string | null>(null);
  const [transcript, setTranscript] = useState('');
  const [isReadingQuestion, setIsReadingQuestion] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const [autoMode, setAutoMode] = useState(true);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<Array<{
    url: string;
    fileName: string;
    fileType: string;
    questionId?: number;
  }>>([]);

  // Reference for speech recognition
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // References for silence detection
  const silenceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const lastTranscriptRef = useRef<string>('');
  const silenceThresholdMs = 2000; // 2 seconds of silence to auto-stop

  const currentQuestion = accidentQuestions[currentQuestionIndex];

  // Function to set up speech recognition event handlers
  const setupSpeechRecognitionHandlers = () => {
    if (!recognitionRef.current) return;

    recognitionRef.current.onresult = (event) => {
      try {
        const transcript = Array.from(event.results)
          .map(result => result[0])
          .map(result => result.transcript)
          .join('');

        setTranscript(transcript);

        // Reset silence detection timer whenever we get new speech
        if (silenceTimeoutRef.current) {
          clearTimeout(silenceTimeoutRef.current);
        }

        // If the transcript has changed, start a new silence detection timer
        if (transcript !== lastTranscriptRef.current) {
          lastTranscriptRef.current = transcript;

          // Set a timeout to detect silence
          silenceTimeoutRef.current = setTimeout(() => {
            if (isListening && autoMode) {
              console.log('Silence detected, stopping listening automatically');
              stopListening();
            }
          }, silenceThresholdMs);
        }
      } catch (error) {
        console.error('Error processing speech recognition result:', error);
      }
    };

    recognitionRef.current.onerror = (event) => {
      console.error('Speech recognition error', event.error);

      // Only handle the error if we're still in listening mode
      if (isListening) {
        setIsListening(false);

        // Don't show error toast for 'aborted' errors as these are often just from stopping recognition
        if (event.error !== 'aborted') {
          toast({
            title: "Speech Recognition Error",
            description: "There was a problem with speech recognition. Please try again.",
          });

          // If in auto mode, try again after a delay
          if (autoMode && !isComplete) {
            setTimeout(() => {
              if (!isListening) {
                startListening();
              }
            }, 1000);
          }
        }
      }
    };

    recognitionRef.current.onend = () => {
      console.log('Speech recognition ended');

      // Clear any pending silence detection
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
        silenceTimeoutRef.current = null;
      }

      // Only process the end event if we're still in listening mode
      // This prevents circular calls between onend and stopListening
      if (isListening) {
        console.log('Recognition ended while still in listening mode');

        // Update state
        setIsListening(false);

        // If we have a transcript and we're in auto mode, process it
        const currentTranscript = transcript;
        if (currentTranscript.trim() && autoMode) {
          // Process the response with a slight delay to ensure state is updated
          setTimeout(() => {
            processResponse(currentTranscript);
          }, 100);
        }
      }
    };
  };

  useEffect(() => {
    // Generate a session ID when the component mounts
    setSessionId(uuidv4());

    // Initialize responses with stored answers
    const initialResponses: Record<number, string> = {};
    accidentQuestions.forEach(q => {
      if (q.answer) {
        initialResponses[q.id] = q.answer;
      }
    });

    setResponses(initialResponses);

    if (accidentQuestions.length > 0) {
      setActiveCategory(accidentQuestions[0].category);
    }

    // Initialize speech recognition
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      // Set up event handlers
      setupSpeechRecognitionHandlers();
    }

    // Create an audio element for text-to-speech
    audioRef.current = new Audio();

    // Read the first question when component mounts
    if (currentQuestion) {
      setTimeout(() => {
        readQuestion(currentQuestion.text);
      }, 1000);
    }

    return () => {
      // Cleanup
      if (audioRef.current) {
        audioRef.current.pause();
      }

      // Stop any ongoing speech synthesis
      window.speechSynthesis.cancel();

      // Stop any ongoing speech recognition
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          console.error('Error stopping speech recognition:', e);
        }
      }

      // Clear any pending silence detection
      if (silenceTimeoutRef.current) {
        clearTimeout(silenceTimeoutRef.current);
        silenceTimeoutRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    // Read the current question whenever it changes
    if (currentQuestion && !isComplete) {
      readQuestion(currentQuestion.text);
    }
  }, [currentQuestionIndex]);

  const readQuestion = async (text: string) => {
    try {
      // Cancel any ongoing speech synthesis
      window.speechSynthesis.cancel();

      // Stop any ongoing speech recognition before starting text-to-speech
      if (isListening && recognitionRef.current) {
        try {
          recognitionRef.current.stop();
          setIsListening(false);
        } catch (e) {
          console.error('Error stopping speech recognition:', e);
        }
      }

      setIsReadingQuestion(true);

      // Create a simple speech synthesis utterance
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;

      // Use browser's built-in speech synthesis
      window.speechSynthesis.speak(utterance);

      utterance.onend = () => {
        setIsReadingQuestion(false);

        // Automatically start listening after the question is read if in auto mode
        if (autoMode && !isComplete) {
          // Add a small delay to make the interaction feel more natural
          setTimeout(() => {
            // Double-check we're not already listening before starting
            if (!isListening) {
              startListening();
            } else {
              console.log('Already listening, not starting again');
            }
          }, 500);
        }
      };

    } catch (error) {
      console.error('Error reading question:', error);
      setIsReadingQuestion(false);
      toast({
        title: "Text-to-Speech Error",
        description: "There was a problem reading the question. Please try again.",
        variant: "destructive",
      });
    }
  };

  const startListening = () => {
    if (!recognitionRef.current) {
      toast({
        title: "Speech Recognition Not Supported",
        description: "Your browser doesn't support speech recognition. Please use a different browser.",
      });
      return;
    }

    // Check if we're already listening - prevent "recognition has already started" error
    if (isListening) {
      console.log('Speech recognition is already active, not starting again');
      return;
    }

    // Reset transcript and silence detection state
    setTranscript('');
    lastTranscriptRef.current = '';
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    // Create a new instance of SpeechRecognition each time to avoid state conflicts
    try {
      // First, set the state to listening
      setIsListening(true);

      // Recreate the recognition object to ensure a clean state
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;

      // Set up event handlers
      setupSpeechRecognitionHandlers();

      // Start recognition
      recognitionRef.current.start();

      toast({
        title: "Listening...",
        description: "Please speak clearly into your microphone.",
      });
    } catch (error) {
      console.error('Error starting speech recognition:', error);
      setIsListening(false);

      toast({
        title: "Speech Recognition Error",
        description: "There was a problem with speech recognition. Please try again.",
        variant: "destructive",
      });

      // If in auto mode, try again after a delay
      if (autoMode && !isComplete) {
        setTimeout(() => {
          // Only try again if we're still not listening
          if (!isListening) {
            console.log('Retrying speech recognition...');
            startListening();
          }
        }, 1000);
      }
    }
  };

  // Function to process a response and move to the next question
  const processResponse = async (answer: string) => {
    if (!answer.trim()) return;

    // Store the response locally
    setResponses(prev => ({
      ...prev,
      [currentQuestion.id]: answer
    }));

    // Store in our storage util
    storeResponse('voice', {
      question: currentQuestion.text,
      answer
    });

    try {
      // Store in Supabase
      await supabase.from('user_responses').insert({
        session_id: sessionId,
        question_id: currentQuestion.id,
        question: currentQuestion.text,
        answer: answer,
        category: currentQuestion.category,
        response_type: 'voice'
      });
    } catch (error) {
      console.error('Error storing response in Supabase:', error);
    }

    toast({
      title: "Response Recorded",
      description: "We've recorded your answer.",
    });

    // Reset transcript
    setTranscript('');

    // Automatically move to the next question if in auto mode
    if (autoMode) {
      // Show a brief "processing" message
      toast({
        title: "Processing...",
        description: "Moving to the next question",
        duration: 1000,
      });

      setTimeout(() => {
        handleNext();
      }, 1500);
    }
  };

  const stopListening = async () => {
    // If we're not listening, don't do anything
    if (!isListening) return;

    // Set state to not listening first to prevent any race conditions
    setIsListening(false);

    // Then try to stop the recognition
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch (error) {
        console.error('Error stopping speech recognition:', error);
        // Even if there's an error, we'll continue processing the transcript
      }
    }

    // Clear any pending silence detection
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    // Use transcript if available or fallback to pre-defined answer
    const answer = transcript || currentQuestion.answer || '';

    // Only proceed if we have an actual answer
    if (answer.trim()) {
      // Process the response
      await processResponse(answer);
    } else {
      // If no answer was provided, restart listening
      if (autoMode && !isComplete) {
        toast({
          title: "No response detected",
          description: "Please try speaking again.",
        });

        // Make sure we're not already listening before trying to start again
        setTimeout(() => {
          if (!isListening) {
            startListening();
          }
        }, 1000);
      }
    }
  };

  const handleNext = () => {
    // Stop any ongoing speech synthesis
    window.speechSynthesis.cancel();

    // Stop any ongoing listening
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
    }

    // Clear any pending silence detection
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    if (currentQuestionIndex < accidentQuestions.length - 1) {
      const nextIndex = currentQuestionIndex + 1;
      setCurrentQuestionIndex(nextIndex);
      setActiveCategory(accidentQuestions[nextIndex].category);

      // The useEffect will trigger the reading of the next question
    } else {
      setIsComplete(true);

      toast({
        title: "Report Complete",
        description: "Thank you for providing all the details about your accident.",
      });
    }
  };

  const handlePrevious = () => {
    // Stop any ongoing speech synthesis
    window.speechSynthesis.cancel();

    // Stop any ongoing listening
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
    }

    // Clear any pending silence detection
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    if (currentQuestionIndex > 0) {
      const prevIndex = currentQuestionIndex - 1;
      setCurrentQuestionIndex(prevIndex);
      setActiveCategory(accidentQuestions[prevIndex].category);

      // The useEffect will trigger the reading of the previous question
    }
  };

  const resetForm = () => {
    // Stop any ongoing speech synthesis
    window.speechSynthesis.cancel();

    // Stop any ongoing listening
    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
    }

    // Clear any pending silence detection
    if (silenceTimeoutRef.current) {
      clearTimeout(silenceTimeoutRef.current);
      silenceTimeoutRef.current = null;
    }

    setCurrentQuestionIndex(0);
    setResponses({});
    setIsComplete(false);
    setSessionId(uuidv4());
    if (accidentQuestions.length > 0) {
      setActiveCategory(accidentQuestions[0].category);
    }

    // The useEffect will trigger the reading of the first question
  };

  // Handle file upload completion
  const handleFileUploadComplete = (fileUrl: string, fileName: string, fileType: string) => {
    // Add the file to the uploaded files list
    setUploadedFiles(prev => [
      ...prev,
      {
        url: fileUrl,
        fileName,
        fileType,
        questionId: currentQuestion?.id
      }
    ]);

    // Add a reference to the file in the response
    const fileResponse = `[File uploaded: ${fileName}]`;
    setResponses(prev => ({
      ...prev,
      [currentQuestion.id]: prev[currentQuestion.id]
        ? `${prev[currentQuestion.id]}\n${fileResponse}`
        : fileResponse
    }));

    // Hide the file upload UI after successful upload
    setShowFileUpload(false);

    // If in auto mode, move to the next question after a delay
    if (autoMode) {
      setTimeout(() => {
        handleNext();
      }, 1500);
    }
  };

  // Toggle file upload UI
  const toggleFileUpload = () => {
    setShowFileUpload(prev => !prev);
  };

  const hasCurrentResponse = responses[currentQuestion?.id];

  // Get unique categories
  const categories = Array.from(new Set(accidentQuestions.map(q => q.category)));

  // Get current category progress
  const getCurrentCategoryProgress = () => {
    if (!activeCategory) return { current: 0, total: 0 };

    const categoryQuestions = accidentQuestions.filter(q => q.category === activeCategory);
    const currentCatIndex = categoryQuestions.findIndex(q => q.id === currentQuestion.id);

    return {
      current: currentCatIndex + 1,
      total: categoryQuestions.length
    };
  };

  // Get files for current question
  const getCurrentQuestionFiles = () => {
    return uploadedFiles.filter(file => file.questionId === currentQuestion?.id);
  };

  const progress = getCurrentCategoryProgress();

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1">
        <div className="container mx-auto px-4 py-12">
          <div className="max-w-3xl mx-auto">
            <h1 className="text-3xl font-bold mb-8 text-center text-gray-900">Voice Accident Assistant</h1>

            <div className="mb-6">
              <div className="flex flex-wrap gap-2 justify-center">
                {categories.map((category, index) => (
                  <div
                    key={index}
                    className={`px-3 py-1.5 rounded-full text-sm font-medium cursor-pointer transition-colors
                      ${activeCategory === category ? 'bg-brand-600 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
                  >
                    {category}
                  </div>
                ))}
              </div>
            </div>

            <Card className="shadow-md border-gray-200">
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle>Accident Details Collection</CardTitle>
                    <CardDescription>
                      Please answer the following questions about your accident using your voice.
                    </CardDescription>
                    {activeCategory && (
                      <div className="mt-2 text-sm text-gray-600">
                        {activeCategory} - Question {progress.current} of {progress.total}
                      </div>
                    )}
                  </div>
                  <div className="flex items-center space-x-2">
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <div className="flex items-center space-x-2">
                            <Switch
                              id="auto-mode"
                              checked={autoMode}
                              onCheckedChange={setAutoMode}
                            />
                            <label
                              htmlFor="auto-mode"
                              className="text-sm font-medium flex items-center cursor-pointer"
                            >
                              {autoMode ? (
                                <>
                                  <Zap className="h-4 w-4 text-brand-600 mr-1" />
                                  <span>Auto Mode (Hands-free)</span>
                                </>
                              ) : (
                                <>
                                  <ZapOff className="h-4 w-4 text-gray-500 mr-1" />
                                  <span className="text-gray-500">Manual Mode</span>
                                </>
                              )}
                            </label>
                          </div>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>
                            {autoMode
                              ? "Fully automated: questions will be read aloud and the system will automatically progress after you speak"
                              : "Manual control: you'll need to click buttons to progress through questions"}
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                </div>
              </CardHeader>

              <CardContent>
                {!isComplete ? (
                  <>
                    <div className="bg-gray-50 p-6 rounded-lg mb-6">
                      <div className="flex items-start gap-4">
                        <div className={`p-2 rounded-full ${isReadingQuestion ? 'bg-brand-100 animate-pulse' : 'bg-brand-100'}`}>
                          <Volume2 className="h-5 w-5 text-brand-600" />
                        </div>
                        <div className="flex-1">
                          <p className="text-lg font-medium">{currentQuestion.text}</p>
                          {transcript && isListening && (
                            <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200 animate-pulse">
                              <p className="text-gray-700">{transcript}</p>
                            </div>
                          )}
                          {hasCurrentResponse && !isListening && (
                            <div className="mt-4 p-3 bg-white rounded-lg border border-gray-200">
                              <p className="text-gray-700">{responses[currentQuestion.id]}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-col gap-4">
                      {/* Current question files */}
                      {getCurrentQuestionFiles().length > 0 && (
                        <div className="mb-4 space-y-2">
                          <p className="text-sm font-medium text-gray-700">Uploaded files:</p>
                          {getCurrentQuestionFiles().map((file, index) => (
                            <FileDisplay
                              key={index}
                              fileUrl={file.url}
                              fileName={file.fileName}
                              fileType={file.fileType}
                            />
                          ))}
                        </div>
                      )}

                      {/* File upload UI */}
                      {showFileUpload ? (
                        <div className="mb-4">
                          <FileUpload
                            sessionId={sessionId}
                            onUploadComplete={handleFileUploadComplete}
                            questionId={currentQuestion?.id}
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
                        <div className="flex justify-center gap-4">
                          <Button
                            size="lg"
                            onClick={isListening ? stopListening : startListening}
                            className={`rounded-full h-16 w-16 ${isListening ? 'bg-red-500 hover:bg-red-600' : 'bg-brand-600 hover:bg-brand-700'}`}
                            disabled={!('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) || isReadingQuestion}
                          >
                            {isListening ? (
                              <MicOff className="h-6 w-6" />
                            ) : (
                              <Mic className="h-6 w-6" />
                            )}
                          </Button>

                          <Button
                            size="icon"
                            variant="outline"
                            onClick={toggleFileUpload}
                            className="rounded-full h-12 w-12"
                            title="Upload a file"
                          >
                            <Upload className="h-5 w-5" />
                          </Button>
                        </div>
                      )}

                      <div className="text-center text-sm text-gray-500">
                        {isListening
                          ? (autoMode ? 'Recording... (will stop automatically)' : 'Click to stop recording')
                          : (autoMode ? 'Recording will start automatically' : 'Click to start recording your answer or upload a file')
                        }
                      </div>

                      {/* Only show navigation buttons in manual mode */}
                      {!autoMode && (
                        <div className="flex justify-between mt-4">
                          <Button variant="outline" onClick={handlePrevious} disabled={currentQuestionIndex === 0}>
                            <ChevronLeft className="mr-1 h-4 w-4" />
                            Previous
                          </Button>
                          <Button onClick={handleNext} disabled={!hasCurrentResponse && !transcript && getCurrentQuestionFiles().length === 0}>
                            {currentQuestionIndex === accidentQuestions.length - 1 ? 'Complete' : 'Next'}
                            {currentQuestionIndex !== accidentQuestions.length - 1 && (
                              <ChevronRight className="ml-1 h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      )}

                      {/* Show auto-progress indicator in auto mode */}
                      {autoMode && hasCurrentResponse && (
                        <div className="text-center mt-4 text-sm text-gray-600">
                          <p>Automatically proceeding to next question...</p>
                        </div>
                      )}
                    </div>
                  </>
                ) : (
                  <div className="py-6">
                    <div className="flex flex-col items-center justify-center text-center gap-4">
                      <div className="h-16 w-16 bg-green-100 rounded-full flex items-center justify-center">
                        <svg className="h-8 w-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                      <h3 className="text-xl font-semibold text-gray-900">Accident Report Complete</h3>
                      <p className="text-gray-600 max-w-md">
                        Thank you for providing all the necessary details about your accident. Your case number is #AC-2023-8954.
                      </p>
                      <div className="bg-amber-50 border border-amber-200 p-4 rounded-lg flex items-start gap-3 w-full mt-4">
                        <AlertCircle className="h-5 w-5 text-amber-500 flex-shrink-0 mt-0.5" />
                        <p className="text-sm text-amber-800">
                          An insurance representative will contact you within 24 hours to discuss the next steps in your claim process.
                        </p>
                      </div>
                      <Button onClick={resetForm} className="mt-4">
                        Start a New Report
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default VoiceAgent;
