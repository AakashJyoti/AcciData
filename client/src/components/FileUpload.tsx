import React, { useState, useRef } from 'react';
import { Upload, X, File, Check, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/hooks/use-toast';

interface FileUploadProps {
  sessionId: string;
  onUploadComplete?: (fileUrl: string, fileName: string, fileType: string) => void;
  questionId?: number;
}

type FileStatus = 'idle' | 'uploading' | 'success' | 'error';

interface FileItem {
  file: File;
  progress: number;
  status: FileStatus;
  url?: string;
  error?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({ sessionId, onUploadComplete, questionId }) => {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFiles(e.target.files);
    }
  };

  const handleFiles = (fileList: FileList) => {
    const newFiles = Array.from(fileList).map(file => ({
      file,
      progress: 0,
      status: 'idle' as FileStatus
    }));
    
    setFiles(prev => [...prev, ...newFiles]);
    
    // Upload each file
    newFiles.forEach(fileItem => {
      uploadFile(fileItem.file);
    });
  };

  const uploadFile = async (file: File) => {
    // Update file status to uploading
    setFiles(prev => 
      prev.map(f => f.file === file ? { ...f, status: 'uploading' as FileStatus } : f)
    );

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
            setFiles(prev => 
              prev.map(f => f.file === file ? { ...f, progress: percent } : f)
            );
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

      // Update file status to success
      setFiles(prev => 
        prev.map(f => f.file === file ? { 
          ...f, 
          status: 'success', 
          url: publicUrl 
        } : f)
      );

      toast({
        title: "File Uploaded",
        description: `${file.name} has been uploaded successfully.`,
      });

      // Call the callback if provided
      if (onUploadComplete) {
        onUploadComplete(publicUrl, file.name, file.type);
      }

    } catch (error) {
      console.error('Error uploading file:', error);
      
      // Update file status to error
      setFiles(prev => 
        prev.map(f => f.file === file ? { 
          ...f, 
          status: 'error', 
          error: error.message || 'Failed to upload file' 
        } : f)
      );

      toast({
        title: "Upload Failed",
        description: `Failed to upload ${file.name}. Please try again.`,
        variant: "destructive",
      });
    }
  };

  const removeFile = (file: File) => {
    setFiles(prev => prev.filter(f => f.file !== file));
  };

  return (
    <div className="w-full">
      <div 
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragging ? 'border-brand-600 bg-brand-50' : 'border-gray-300 hover:border-brand-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileInputChange}
          className="hidden"
          multiple
          accept="image/*,application/pdf,.doc,.docx"
        />
        <Upload className="h-10 w-10 mx-auto text-gray-400" />
        <p className="mt-2 text-sm font-medium text-gray-700">
          Drag and drop files here, or click to select files
        </p>
        <p className="mt-1 text-xs text-gray-500">
          Supports images, PDFs, and documents up to 10MB
        </p>
      </div>

      {files.length > 0 && (
        <div className="mt-4 space-y-3">
          {files.map((fileItem, index) => (
            <div key={index} className="flex items-center p-3 bg-gray-50 rounded-lg">
              <div className="mr-3">
                <File className="h-8 w-8 text-gray-400" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {fileItem.file.name}
                </p>
                <p className="text-xs text-gray-500">
                  {(fileItem.file.size / 1024).toFixed(1)} KB
                </p>
                {fileItem.status === 'uploading' && (
                  <Progress value={fileItem.progress} className="h-1 mt-1" />
                )}
                {fileItem.status === 'error' && (
                  <p className="text-xs text-red-500 mt-1">{fileItem.error}</p>
                )}
              </div>
              <div className="ml-3 flex-shrink-0">
                {fileItem.status === 'success' ? (
                  <Check className="h-5 w-5 text-green-500" />
                ) : fileItem.status === 'error' ? (
                  <AlertCircle className="h-5 w-5 text-red-500" />
                ) : (
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeFile(fileItem.file);
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
