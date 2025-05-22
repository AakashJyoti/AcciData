
// Simple storage utility for storing responses from chat and voice interactions
type ResponseType = 'chat' | 'voice';

interface ResponseData {
  question: string;
  answer: string;
  timestamp?: number;
}

// Store a response in localStorage
export const storeResponse = (type: ResponseType, data: ResponseData) => {
  try {
    // Get existing responses or initialize empty array
    const storageKey = `accidata_${type}_responses`;
    const existingResponses = localStorage.getItem(storageKey);
    const responses = existingResponses ? JSON.parse(existingResponses) : [];
    
    // Add timestamp to the data
    const responseWithTimestamp = {
      ...data,
      timestamp: Date.now()
    };
    
    // Add new response to array
    responses.push(responseWithTimestamp);
    
    // Store updated responses
    localStorage.setItem(storageKey, JSON.stringify(responses));
    
    console.log(`Stored ${type} response:`, responseWithTimestamp);
    return true;
  } catch (error) {
    console.error(`Error storing ${type} response:`, error);
    return false;
  }
};

// Get all responses of a specific type
export const getResponses = (type: ResponseType): ResponseData[] => {
  try {
    const storageKey = `accidata_${type}_responses`;
    const responses = localStorage.getItem(storageKey);
    return responses ? JSON.parse(responses) : [];
  } catch (error) {
    console.error(`Error retrieving ${type} responses:`, error);
    return [];
  }
};

// Clear all responses of a specific type
export const clearResponses = (type: ResponseType): boolean => {
  try {
    const storageKey = `accidata_${type}_responses`;
    localStorage.removeItem(storageKey);
    return true;
  } catch (error) {
    console.error(`Error clearing ${type} responses:`, error);
    return false;
  }
};
