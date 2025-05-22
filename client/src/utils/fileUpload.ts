import { supabase } from '@/integrations/supabase/client';

export interface UploadProgressCallback {
  (progress: number): void;
}

export interface UploadResult {
  success: boolean;
  url?: string;
  error?: string;
  fileName?: string;
  fileType?: string;
}

/**
 * Uploads a file to Supabase storage and records metadata in the documents table
 * 
 * @param file The file to upload
 * @param sessionId The current session ID
 * @param onProgress Optional callback for upload progress
 * @returns Promise with upload result
 */
export const uploadFile = async (
  file: File, 
  sessionId: string,
  onProgress?: UploadProgressCallback
): Promise<UploadResult> => {
  try {
    // Create a unique file name
    const fileExt = file.name.split('.').pop();
    const fileName = `${Math.random().toString(36).substring(2, 15)}_${Date.now()}.${fileExt}`;
    const filePath = `${sessionId}/${fileName}`;

    // Upload to Supabase Storage
    const { data, error } = await supabase.storage
      .from('accident-documents')
      .upload(filePath, file, {
        cacheControl: '3600',
        upsert: false,
        onUploadProgress: (progress) => {
          const percent = Math.round((progress.loaded / progress.total) * 100);
          if (onProgress) {
            onProgress(percent);
          }
        }
      });

    if (error) {
      throw error;
    }

    // Get the public URL
    const { data: { publicUrl } } = supabase.storage
      .from('accident-documents')
      .getPublicUrl(filePath);

    // Store file metadata in the documents table
    const { error: insertError } = await supabase
      .from('documents')
      .insert({
        session_id: sessionId,
        filename: file.name,
        file_url: publicUrl,
        file_type: file.type
      });

    if (insertError) {
      throw insertError;
    }

    return {
      success: true,
      url: publicUrl,
      fileName: file.name,
      fileType: file.type
    };
  } catch (error) {
    console.error('Error uploading file:', error);
    return {
      success: false,
      error: error.message || 'Failed to upload file'
    };
  }
};

/**
 * Gets all files uploaded for a specific session
 * 
 * @param sessionId The session ID to get files for
 * @returns Promise with array of file data
 */
export const getSessionFiles = async (sessionId: string) => {
  try {
    const { data, error } = await supabase
      .from('documents')
      .select('*')
      .eq('session_id', sessionId);
    
    if (error) {
      throw error;
    }
    
    return data;
  } catch (error) {
    console.error('Error fetching session files:', error);
    return [];
  }
};

/**
 * Validates if a file is acceptable for upload
 * 
 * @param file The file to validate
 * @returns Object with validation result
 */
export const validateFile = (file: File): { valid: boolean; message?: string } => {
  // Check file size (10MB max)
  const maxSize = 10 * 1024 * 1024; // 10MB
  if (file.size > maxSize) {
    return {
      valid: false,
      message: `File size exceeds 10MB limit (${(file.size / (1024 * 1024)).toFixed(2)}MB)`
    };
  }

  // Check file type
  const allowedTypes = [
    'image/jpeg',
    'image/png',
    'image/gif',
    'application/pdf',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
  ];
  
  if (!allowedTypes.includes(file.type)) {
    return {
      valid: false,
      message: 'File type not supported. Please upload images, PDFs, or documents.'
    };
  }

  return { valid: true };
};
