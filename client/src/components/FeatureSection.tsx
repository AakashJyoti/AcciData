
import React from 'react';
import { MessageCircle, Mic, BarChart3, Shield } from "lucide-react";

const features = [
  {
    icon: <MessageCircle className="h-10 w-10 text-brand-600" />,
    title: "Interactive Chat",
    description: "Our AI-powered chat interface guides you through the accident reporting process with simple, conversational questions."
  },
  {
    icon: <Mic className="h-10 w-10 text-brand-600" />,
    title: "Voice Assistant",
    description: "Record your accident details by simply speaking to our voice agent - perfect for when you're on the go."
  },
  {
    icon: <BarChart3 className="h-10 w-10 text-brand-600" />,
    title: "Smart Analytics",
    description: "Get insights into accident patterns and risk factors based on collected data and AI analysis."
  },
  {
    icon: <Shield className="h-10 w-10 text-brand-600" />,
    title: "Secure Storage",
    description: "All your accident data is stored securely with enterprise-grade encryption and privacy controls."
  },
];

const FeatureSection: React.FC = () => {
  return (
    <section id="features" className="py-20 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Powerful Features</h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Our platform offers cutting-edge tools for accident reporting and data collection.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="bg-white p-6 rounded-lg border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="mb-5">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-900">{feature.title}</h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
        
        <div className="mt-20" id="how-it-works">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Our platform simplifies accident reporting in just a few easy steps.
            </p>
          </div>
          
          <div className="flex flex-col md:flex-row gap-8">
            <div className="flex-1 relative">
              <div className="border border-gray-200 rounded-lg p-6 shadow-sm bg-white">
                <div className="flex items-center mb-4">
                  <div className="h-8 w-8 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center font-bold mr-3">1</div>
                  <h3 className="text-xl font-semibold">Choose Your Interface</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  Select between our chat or voice assistant based on your preference and situation.
                </p>
              </div>
            </div>
            
            <div className="flex-1 relative">
              <div className="border border-gray-200 rounded-lg p-6 shadow-sm bg-white">
                <div className="flex items-center mb-4">
                  <div className="h-8 w-8 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center font-bold mr-3">2</div>
                  <h3 className="text-xl font-semibold">Answer Questions</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  Our AI asks you relevant questions about your accident in a conversational manner.
                </p>
              </div>
            </div>
            
            <div className="flex-1 relative">
              <div className="border border-gray-200 rounded-lg p-6 shadow-sm bg-white">
                <div className="flex items-center mb-4">
                  <div className="h-8 w-8 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center font-bold mr-3">3</div>
                  <h3 className="text-xl font-semibold">Get Your Report</h3>
                </div>
                <p className="text-gray-600 mb-4">
                  Receive a comprehensive accident report that can be shared with insurance or authorities.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default FeatureSection;
