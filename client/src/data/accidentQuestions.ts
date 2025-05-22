
export interface Question {
  id: number;
  category: string;
  text: string;
  answer?: string;
}

export const accidentQuestions: Question[] = [
  // Accident Details
  { id: 1, category: "🛠️ Accident Details", text: "When did the accident happen?", answer: "March 12, 2024, at 4:15 PM." },
  { id: 2, category: "🛠️ Accident Details", text: "Where did the accident occur?", answer: "At the intersection of Main Street and 1st Avenue in Los Angeles." },
  { id: 3, category: "🛠️ Accident Details", text: "What type of road was it? (e.g., highway, residential street)", answer: "It was a four-lane highway." },
  { id: 4, category: "🛠️ Accident Details", text: "Was it during day or night?", answer: "It happened during daylight." },
  { id: 5, category: "🛠️ Accident Details", text: "What was the weather like at the time?", answer: "It was raining lightly." },
  { id: 6, category: "🛠️ Accident Details", text: "What were the road conditions?", answer: "The road was slippery due to rain." },
  { id: 7, category: "🛠️ Accident Details", text: "How many vehicles were involved?", answer: "Two cars." },
  { id: 8, category: "🛠️ Accident Details", text: "Were there any pedestrians or cyclists involved?", answer: "No pedestrians or cyclists were involved." },
  { id: 9, category: "🛠️ Accident Details", text: "Was there any construction in the area?", answer: "Yes, there was a construction barrier on the right side of the road." },
  { id: 10, category: "🛠️ Accident Details", text: "Was traffic heavy or light at the time?", answer: "Traffic was moderate." },
  
  // Vehicle & Driver Info
  { id: 11, category: "🚗 Vehicle & Driver Info", text: "What were you driving? (Make, model, year)", answer: "A 2020 Honda Accord." },
  { id: 12, category: "🚗 Vehicle & Driver Info", text: "What type of vehicle is it?", answer: "It's a sedan." },
  { id: 13, category: "🚗 Vehicle & Driver Info", text: "Was your vehicle damaged? If yes, where?", answer: "Yes, the front bumper was smashed." },
  { id: 14, category: "🚗 Vehicle & Driver Info", text: "Who was the other driver and what vehicle were they driving?", answer: "Driver B in a 2018 Ford F-150." },
  { id: 15, category: "🚗 Vehicle & Driver Info", text: "What type of vehicle was the other party driving?", answer: "A pickup truck." },
  { id: 16, category: "🚗 Vehicle & Driver Info", text: "Was either driver using a phone or distracted?", answer: "The other driver was talking on their phone." },
  { id: 17, category: "🚗 Vehicle & Driver Info", text: "Was your seatbelt fastened?", answer: "Yes." },
  { id: 18, category: "🚗 Vehicle & Driver Info", text: "Was the other driver wearing a seatbelt?", answer: "I believe so." },
  { id: 19, category: "🚗 Vehicle & Driver Info", text: "Did either vehicle have mechanical issues?", answer: "No issues that I know of." },
  { id: 20, category: "🚗 Vehicle & Driver Info", text: "Was anyone under the influence of drugs or alcohol?", answer: "No, both drivers were sober." },
  
  // Vehicle Movement & Behaviour
  { id: 21, category: "📊 Vehicle Movement & Behaviour", text: "What direction were you traveling?", answer: "Northbound." },
  { id: 22, category: "📊 Vehicle Movement & Behaviour", text: "What direction was the other vehicle traveling?", answer: "Southbound, turning left." },
  { id: 23, category: "📊 Vehicle Movement & Behaviour", text: "What was your speed at the time?", answer: "About 30 mph." },
  { id: 24, category: "📊 Vehicle Movement & Behaviour", text: "What was the other vehicle's speed?", answer: "Probably around 15 mph during the turn." },
  { id: 25, category: "📊 Vehicle Movement & Behaviour", text: "Were you following the speed limit?", answer: "Yes, the speed limit was 35 mph." },
  { id: 26, category: "📊 Vehicle Movement & Behaviour", text: "Were you changing lanes or turning?", answer: "No, I was going straight." },
  { id: 27, category: "📊 Vehicle Movement & Behaviour", text: "Was the other vehicle changing lanes or turning?", answer: "Yes, it was turning left." },
  { id: 28, category: "📊 Vehicle Movement & Behaviour", text: "Did either vehicle run a red light or stop sign?", answer: "The other car ran a red light." },
  { id: 29, category: "📊 Vehicle Movement & Behaviour", text: "Did you yield when required?", answer: "Yes, I had the right of way." },
  { id: 30, category: "📊 Vehicle Movement & Behaviour", text: "Was anyone reversing or parked?", answer: "No, both vehicles were moving." },
  
  // Collision Specifics
  { id: 31, category: "💥 Collision Specifics", text: "Where did your vehicle get hit?", answer: "In the front." },
  { id: 32, category: "💥 Collision Specifics", text: "Where did the other vehicle get hit?", answer: "On the passenger side." },
  { id: 33, category: "💥 Collision Specifics", text: "What part of the road were you in? (e.g., left lane, right lane)", answer: "I was in the right lane." },
  { id: 34, category: "💥 Collision Specifics", text: "Did airbags deploy in your vehicle?", answer: "Yes, the front airbags deployed." },
  { id: 35, category: "💥 Collision Specifics", text: "Did airbags deploy in the other vehicle?", answer: "I'm not sure." },
  { id: 36, category: "💥 Collision Specifics", text: "Was there any vehicle rollover?", answer: "No." },
  { id: 37, category: "💥 Collision Specifics", text: "Was there secondary impact (e.g., hitting a tree, pole)?", answer: "No, just the other car." },
  { id: 38, category: "💥 Collision Specifics", text: "Was a tow truck needed?", answer: "Yes, both cars were towed." },
  { id: 39, category: "💥 Collision Specifics", text: "Was the accident reported to insurance?", answer: "Yes, I've already contacted mine." },
  { id: 40, category: "💥 Collision Specifics", text: "Was the accident reported to the police?", answer: "Yes, a report was filed." },
  
  // Witnesses & Evidence
  { id: 41, category: "👮‍♂️ Witnesses & Evidence", text: "Were there any witnesses?", answer: "Yes, a pedestrian saw everything." },
  { id: 42, category: "👮‍♂️ Witnesses & Evidence", text: "Do you have the witness contact information?", answer: "Yes, I have their name and phone number." },
  { id: 43, category: "👮‍♂️ Witnesses & Evidence", text: "Did you take photos of the scene?", answer: "Yes, I took photos of both cars and the intersection." },
  { id: 44, category: "👮‍♂️ Witnesses & Evidence", text: "Do you have dashcam or CCTV footage?", answer: "Yes, from my dashcam." },
  { id: 45, category: "👮‍♂️ Witnesses & Evidence", text: "Did the police assign fault at the scene?", answer: "Yes, they cited the other driver." },
  
  // Context & Follow-up
  { id: 46, category: "🧾 Context & Follow-up", text: "Do you have a copy of the police report?", answer: "Yes, I have the report number and document." },
  { id: 47, category: "🧾 Context & Follow-up", text: "Have you spoken to the other driver after the accident?", answer: "Yes, we exchanged information and briefly discussed the crash." },
  { id: 48, category: "🧾 Context & Follow-up", text: "Were there any injuries? If so, to whom?", answer: "No major injuries, just minor bruises for me." },
  { id: 49, category: "🧾 Context & Follow-up", text: "Do you want to provide any additional information?", answer: "I think the other driver was rushing to make the light." },
  { id: 50, category: "🧾 Context & Follow-up", text: "Would you like a summary of the findings once complete?", answer: "Yes, please send it to my email." },
];

// Group questions by category
export const getQuestionsByCategory = () => {
  const categories: Record<string, Question[]> = {};
  
  accidentQuestions.forEach(question => {
    if (!categories[question.category]) {
      categories[question.category] = [];
    }
    categories[question.category].push(question);
  });
  
  return categories;
};
