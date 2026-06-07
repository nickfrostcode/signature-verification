import { AlertCircle, X } from 'lucide-react'
import { useEffect } from 'react'

export default function Toast({ message, onClose }) {
   useEffect(() => {
      if (message) {
         const timer = setTimeout(() => {
            onClose()
         }, 5000)
         return () => clearTimeout(timer)
      }
   }, [message, onClose])

   if (!message) return null

   return (
      <div className="fixed top-6 right-6 z-50 animate-in slide-in-from-top-4 fade-in duration-300 max-w-[90vw] sm:max-w-sm">
         <div className="bg-[#93000a] border border-[#ffb4ab]/50 text-[#ffdad6] px-4 py-3 rounded-lg shadow-xl flex items-center gap-3">
            <AlertCircle className="shrink-0" size={16} />
            <p className="text-xs sm:text-sm font-medium flex-1 line-clamp-2 leading-tight">{message}</p>
            <button
               onClick={onClose}
               className="text-[#ffdad6]/80 hover:text-white transition-colors shrink-0 -mr-1 p-1 rounded-md hover:bg-black/10"
            >
               <X size={14} />
            </button>
         </div>
      </div>
   )
}
