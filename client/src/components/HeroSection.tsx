
import React from 'react';
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";

const HeroSection: React.FC = () => {
  return (
    <div className="relative overflow-hidden bg-white">
      <div className="gradient-bg absolute inset-0 z-0"></div>
      <div className="container mx-auto px-4 pt-16 pb-24 relative z-10">
        <div className="flex flex-col md:flex-row gap-12 items-center">
          <div className="flex-1 max-w-2xl">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-gray-900 mb-6">
              Accident Details Collection{" "}
              <span className="text-brand-600">Made Simple</span>
            </h1>
            <p className="text-xl text-gray-600 mb-8 leading-relaxed">
              Our AI-powered platform streamlines accident reporting with intuitive chat and voice interfaces. Get accurate information quickly and efficiently.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button size="lg" asChild>
                <Link to="/voice-agent">Try Voice Assistant</Link>
              </Button>
              <Button size="lg" variant="outline">
                Learn More
              </Button>
            </div>
          </div>
          <div className="flex-1 max-w-md relative">
            <div className="relative bg-white rounded-lg shadow-xl p-6 border border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-lg">Accident Assistant</h3>
                <span className="flex items-center text-xs text-green-600">
                  <span className="h-2 w-2 rounded-full bg-green-500 mr-1"></span>
                  Online
                </span>
              </div>
              <div className="space-y-3">
                <div className="bg-brand-50 p-3 rounded-lg rounded-tl-none max-w-[80%]">
                  <p className="text-sm">Hello! I'm here to help you report your accident. Can you tell me when it happened?</p>
                </div>
                <div className="bg-gray-100 p-3 rounded-lg rounded-tr-none max-w-[80%] ml-auto">
                  <p className="text-sm">It happened yesterday around 2pm at Main Street.</p>
                </div>
                <div className="bg-brand-50 p-3 rounded-lg rounded-tl-none max-w-[80%]">
                  <p className="text-sm">Thank you. Were there any injuries or other vehicles involved?</p>
                </div>
                <div className="animate-pulse-subtle mt-2">
                  <div className="h-3 w-10 rounded bg-gray-200"></div>
                </div>
              </div>
            </div>
            <div className="absolute -z-10 bg-accent2-400 w-full h-full rounded-lg top-6 left-6 opacity-20"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;
