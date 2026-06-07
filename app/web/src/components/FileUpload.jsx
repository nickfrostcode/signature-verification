import { useMemo } from 'react'
import { UploadCloud, Image as ImageIcon } from 'lucide-react'

export default function FileUpload({ label, file, setFile }) {
  const previewUrl = useMemo(() => {
    if (!file) return null
    return URL.createObjectURL(file)
  }, [file])

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-semibold text-on-surface-muted uppercase tracking-wider">
        {label}
      </label>
      <div className="relative group">
        <input
          type="file"
          accept="image/*"
          onChange={handleChange}
          title="Click or drag and drop to upload"
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
        />
        <div className={`relative overflow-hidden flex items-center justify-center border-2 border-dashed rounded-lg h-32 transition-all duration-200 ease-in-out group-hover:border-primary group-hover:bg-primary-dark/20 ${file ? 'border-primary bg-primary-dark/30 shadow-[0_0_15px_rgba(var(--color-primary-rgb),0.1)]' : 'border-border bg-surface-dim'}`}>
          {previewUrl ? (
            <>
              <div className="absolute inset-0 w-full h-full p-2">
                <img 
                  src={previewUrl} 
                  alt="Preview" 
                  className="w-full h-full object-contain bg-white rounded shadow"
                />
              </div>
              <div className="absolute inset-0 bg-surface-lowest/80 backdrop-blur-sm flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-all duration-300">
                <UploadCloud className="text-primary mb-2 transform scale-75 group-hover:scale-100 transition-transform duration-300" size={28} />
                <p className="text-white font-medium text-sm">Drop to Replace</p>
                <p className="text-on-surface-muted text-xs mt-1">or click to browse</p>
              </div>
            </>
          ) : (
            <div className="text-center text-on-surface-muted px-4 group-hover:text-primary transition-colors">
              <ImageIcon className="mx-auto mb-2 opacity-50 group-hover:opacity-100 transition-opacity transform group-hover:-translate-y-1 duration-300" size={28} />
              <span className="text-sm font-medium">Click or drag image</span>
              <p className="text-xs opacity-70 mt-1">PNG, JPG up to 10MB</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
