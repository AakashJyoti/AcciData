import React from 'react';
import { File, Image, FileText, Download, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface FileDisplayProps {
  fileUrl: string;
  fileName: string;
  fileType: string;
  showDownload?: boolean;
}

const FileDisplay: React.FC<FileDisplayProps> = ({ 
  fileUrl, 
  fileName, 
  fileType,
  showDownload = true
}) => {
  const isImage = fileType.startsWith('image/');
  const isPdf = fileType === 'application/pdf';
  const isDocument = fileType.includes('word') || fileType.includes('document');

  const getFileIcon = () => {
    if (isImage) return <Image className="h-5 w-5" />;
    if (isPdf) return <FileText className="h-5 w-5" />;
    if (isDocument) return <FileText className="h-5 w-5" />;
    return <File className="h-5 w-5" />;
  };

  const getFileExtension = () => {
    if (fileName.includes('.')) {
      return fileName.split('.').pop()?.toUpperCase();
    }
    return '';
  };

  const handleDownload = () => {
    window.open(fileUrl, '_blank');
  };

  return (
    <div className="flex items-center p-3 bg-gray-50 rounded-lg border border-gray-200">
      <div className="flex items-center justify-center h-10 w-10 rounded-md bg-gray-200 text-gray-700">
        {getFileIcon()}
      </div>
      
      <div className="ml-3 flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {fileName}
        </p>
        <p className="text-xs text-gray-500">
          {getFileExtension()}
        </p>
      </div>
      
      {showDownload && (
        <Button
          variant="ghost"
          size="icon"
          onClick={handleDownload}
          title="Download file"
        >
          {isImage ? (
            <ExternalLink className="h-4 w-4" />
          ) : (
            <Download className="h-4 w-4" />
          )}
        </Button>
      )}
      
      {isImage && (
        <div className="mt-2 w-full">
          <img 
            src={fileUrl} 
            alt={fileName} 
            className="max-h-40 rounded-md object-contain mx-auto"
          />
        </div>
      )}
    </div>
  );
};

export default FileDisplay;
