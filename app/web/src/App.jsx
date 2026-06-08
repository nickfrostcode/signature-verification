import { useState, Activity, useRef } from 'react'
import axios from 'axios'
import { Cpu, Fingerprint, XCircle } from 'lucide-react'
import FileUpload from './components/FileUpload'
import ResultsDisplay from './components/ResultsDisplay'
import Toast from './components/Toast'

// Backend API URL from .env
const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

function App() {
   const [mode, setMode] = useState('baseline') // 'baseline' or 'siamese'

   // Baseline Form State
   const [baselineFile, setBaselineFile] = useState(null)
   const [baselineThreshold, setBaselineThreshold] = useState(0.5)

   // Siamese Form State
   const [siameseFileA, setSiameseFileA] = useState(null)
   const [siameseFileB, setSiameseFileB] = useState(null)
   const [siameseThreshold, setSiameseThreshold] = useState(0.345)

   // Request State
   const [loading, setLoading] = useState(false)

   // We keep separate results/errors for each mode so switching tabs remembers them
   const [baselineResult, setBaselineResult] = useState(null)
   const [baselineError, setBaselineError] = useState(null)

   const [siameseResult, setSiameseResult] = useState(null)
   const [siameseError, setSiameseError] = useState(null)

   const abortControllerRef = useRef(null)

   const handleCancel = () => {
      if (abortControllerRef.current) {
         abortControllerRef.current.abort()
         abortControllerRef.current = null
      }
      setLoading(false)
      if (mode === 'baseline') {
         setBaselineError("Execution stopped by user.")
      } else {
         setSiameseError("Execution stopped by user.")
      }
   }

   const handleSubmit = async (e, currentMode) => {
      e.preventDefault()
      
      // If already loading, prevent submission (cancellation handled by Stop button)
      if (loading) return

      setLoading(true)
      
      abortControllerRef.current = new AbortController()

      const formData = new FormData()
      formData.append('explain', true)

      try {
         if (currentMode === 'baseline') {
            setBaselineError(null)
            setBaselineResult(null)
            if (!baselineFile) throw new Error("Please upload a target signature image.")

            formData.append('threshold', baselineThreshold)
            formData.append('image', baselineFile)

            const res = await axios.post(`${API_BASE}/verify/baseline`, formData, {
               signal: abortControllerRef.current.signal
            })
            setBaselineResult(res.data)
         } else {
            setSiameseError(null)
            setSiameseResult(null)
            if (!siameseFileA || !siameseFileB) throw new Error("Please upload both Reference and Questioned signature images.")

            formData.append('threshold', siameseThreshold)
            formData.append('image_a', siameseFileA)
            formData.append('image_b', siameseFileB)

            const res = await axios.post(`${API_BASE}/verify/siamese`, formData, {
               signal: abortControllerRef.current.signal
            })
            setSiameseResult(res.data)
         }
      } catch (err) {
         if (axios.isCancel(err)) {
            const cancelMsg = "Execution stopped by user."
            if (currentMode === 'baseline') setBaselineError(cancelMsg)
            else setSiameseError(cancelMsg)
         } else {
            const errMsg = err.response?.data?.detail || err.message || "An error occurred during verification."
            if (currentMode === 'baseline') setBaselineError(errMsg)
            else setSiameseError(errMsg)
         }
      } finally {
         setLoading(false)
         abortControllerRef.current = null
      }
   }

   return (
      <div className="max-w-6xl mx-auto px-4 py-8 md:py-12">
         <Toast message={mode === 'baseline' ? baselineError : siameseError} onClose={() => mode === 'baseline' ? setBaselineError(null) : setSiameseError(null)} />
         {/* Header */}
         <header className="mb-8 md:mb-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
               <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold tracking-tight text-white mb-2 flex items-center gap-2 sm:gap-3 flex-wrap">
                  <Fingerprint className="text-primary shrink-0" size={32} />
                  <span>Synthetic Intelligence Identity</span>
               </h1>
               <p className="text-on-surface-muted text-base md:text-lg font-medium">Forensic Signature Verification System</p>
            </div>
         </header>

         <main className="grid grid-cols-1 lg:grid-cols-12 gap-8 relative">

            {/* Left Column: Controls */}
            <section className="lg:col-span-4 space-y-6">
               <div className="glass-panel p-6">

                  {/* Mode Toggle */}
                  <div className="flex bg-surface-dim rounded-lg p-1 mb-8 border border-border">
                     <button
                        onClick={() => setMode('baseline')}
                        className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${mode === 'baseline' ? 'bg-border text-white shadow' : 'text-on-surface-muted hover:text-white'}`}
                     >
                        Standard (Single)
                     </button>
                     <button
                        onClick={() => setMode('siamese')}
                        className={`flex-1 py-2 text-sm font-semibold rounded-md transition-all ${mode === 'siamese' ? 'bg-border text-white shadow' : 'text-on-surface-muted hover:text-white'}`}
                     >
                        Comparative (Pair)
                     </button>
                  </div>

                  {/* We use React 19 <Activity> to keep the inactive form in memory without resetting its state/inputs */}
                  <Activity mode={mode === 'baseline' ? 'visible' : 'hidden'}>
                     <form onSubmit={(e) => handleSubmit(e, 'baseline')} className="space-y-6" style={{ display: mode === 'baseline' ? 'block' : 'none' }}>
                        <FileUpload
                           label="Target Signature"
                           file={baselineFile}
                           setFile={setBaselineFile}
                        />

                        <div className="pt-4 border-t border-border">
                           <div className="flex justify-between items-center mb-2">
                              <label className="text-sm font-semibold text-on-surface-muted uppercase tracking-wider">Sensitivity</label>
                              <span className="text-primary font-bold bg-primary-dark px-2 py-0.5 rounded text-xs">{baselineThreshold}</span>
                           </div>
                           <input
                              type="range"
                              min="0.0"
                              max="1.0"
                              step="0.01"
                              value={baselineThreshold}
                              onChange={(e) => setBaselineThreshold(parseFloat(e.target.value))}
                              className="w-full accent-primary cursor-pointer"
                           />
                           <p className="text-[11px] text-on-surface-muted mt-1 leading-snug">Adjusts strictness of forgery detection.</p>
                        </div>

                        <SubmitButton loading={loading} onCancel={handleCancel} />
                     </form>
                  </Activity>

                  <Activity mode={mode === 'siamese' ? 'visible' : 'hidden'}>
                     <form onSubmit={(e) => handleSubmit(e, 'siamese')} className="space-y-6" style={{ display: mode === 'siamese' ? 'block' : 'none' }}>
                        <FileUpload
                           label="Reference Document (A)"
                           file={siameseFileA}
                           setFile={setSiameseFileA}
                        />

                        <FileUpload
                           label="Questioned Document (B)"
                           file={siameseFileB}
                           setFile={setSiameseFileB}
                        />

                        <div className="pt-4 border-t border-border">
                           <div className="flex justify-between items-center mb-2">
                              <label className="text-sm font-semibold text-on-surface-muted uppercase tracking-wider">Sensitivity</label>
                              <span className="text-primary font-bold bg-primary-dark px-2 py-0.5 rounded text-xs">{siameseThreshold}</span>
                           </div>
                           <input
                              type="range"
                              min="0.0"
                              max="1.0"
                              step="0.01"
                              value={siameseThreshold}
                              onChange={(e) => setSiameseThreshold(parseFloat(e.target.value))}
                              className="w-full accent-primary cursor-pointer"
                           />
                           <p className="text-[11px] text-on-surface-muted mt-1 leading-snug">Adjusts strictness of similarity comparison.</p>
                        </div>

                        <SubmitButton loading={loading} onCancel={handleCancel} />
                     </form>
                  </Activity>
               </div>
            </section>

            {/* Right Column: Results & Explanations */}
            <section className="lg:col-span-8 relative">

               <Activity mode={mode === 'baseline' ? 'visible' : 'hidden'}>
                  <div style={{ display: mode === 'baseline' ? 'block' : 'none' }}>
                     <ResultsDisplay
                        result={baselineResult}
                        mode="baseline"
                        loading={loading}
                     />
                  </div>
               </Activity>

               <Activity mode={mode === 'siamese' ? 'visible' : 'hidden'}>
                  <div style={{ display: mode === 'siamese' ? 'block' : 'none' }}>
                     <ResultsDisplay
                        result={siameseResult}
                        mode="siamese"
                        loading={loading}
                     />
                  </div>
               </Activity>

            </section>
         </main>
      </div>
   )
}

function SubmitButton({ loading, onCancel }) {
   if (loading) {
      return (
         <button
            type="button"
            onClick={onCancel}
            className="w-full btn-primary bg-[#93000a] hover:bg-[#ffb4ab] hover:text-[#93000a] border-[#ffb4ab]/30 text-[#ffdad6] flex justify-center items-center gap-2 py-3 mt-4 transition-colors duration-200 shadow-md cursor-pointer"
         >
            <XCircle size={20} />
            Stop Execution
         </button>
      )
   }

   return (
      <button
         type="submit"
         className="w-full btn-primary flex justify-center items-center gap-2 py-3 mt-4 cursor-pointer"
      >
         <Cpu size={20} />
         Verify Signature
      </button>
   )
}

export default App
