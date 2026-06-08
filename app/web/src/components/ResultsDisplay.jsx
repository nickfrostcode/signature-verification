import { ShieldCheck, ShieldAlert, FileSearch, Sparkles } from 'lucide-react'

export default function ResultsDisplay({ result, mode, loading }) {
   const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000"

   if (loading) {
      return (
         <div className="h-full min-h-[400px] glass-panel p-8 flex flex-col items-center justify-center relative overflow-hidden">
            <div className="absolute inset-0 bg-surface-dim/50 backdrop-blur-[2px] z-10 flex flex-col items-center justify-center">
               <div className="animate-pulse flex flex-col items-center text-primary">
                  <Sparkles size={48} className="mb-4 animate-spin-slow" />
                  <h3 className="text-xl font-bold tracking-widest uppercase">Processing Analysis...</h3>
                  <p className="text-sm text-on-surface-muted mt-2 font-medium">Running forensic deep learning models</p>
               </div>
            </div>
         </div>
      )
   }

   if (!result) {
      return (
         <div className="h-full min-h-[400px] glass-panel flex flex-col items-center justify-center text-on-surface-muted p-8 text-center border-dashed">
            <FileSearch size={64} className="mb-4 opacity-20" />
            <h3 className="text-xl text-white mb-2">Awaiting Input</h3>
            <p className="max-w-md text-sm">Upload signature images and click verify to run the neural network analysis. Results and explainability heatmaps will appear here.</p>
         </div>
      )
   }

   const isPositive = result.prediction === 'genuine' || result.prediction === 'match'
   const confidence = (result.probability * 100).toFixed(2)

   // Interpretation of the score
   let scoreMeaning = "";
   if (confidence > 95) scoreMeaning = "Extremely High Confidence"
   else if (confidence > 80) scoreMeaning = "High Confidence"
   else if (confidence > 60) scoreMeaning = "Moderate Confidence"
   else scoreMeaning = "Low Confidence (Borderline)"

   return (
      <div className="space-y-6">
         {/* Verdict Header */}
         <div className='glass-panel p-6'>
            <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
               <div className="space-y-4">
                  <div>
                     <p className="font-semibold text-on-surface-muted text-sm mb-1 uppercase tracking-wider">Final Verification Verdict</p>
                     <div className="flex flex-wrap items-center gap-3">
                        <div className={`flex items-center gap-2 px-4 py-2 rounded-full font-bold text-lg shadow-sm ${isPositive
                              ? 'bg-[#004e5f] text-primary border border-primary/30'
                              : 'bg-[#93000a] text-[#ffb4ab] border border-[#ffb4ab]/30'
                           }`}>
                           {isPositive ? <ShieldCheck size={24} /> : <ShieldAlert size={24} />}
                           {isPositive ? 'VERIFIED MATCH' : 'FORGERY DETECTED'}
                        </div>
                        <span className="font-bold text-2xl text-white flex items-center gap-2">
                           <span className={isPositive ? 'text-primary' : 'text-[#ffb4ab]'}>{confidence}%</span>
                        </span>
                     </div>
                  </div>

                  <div className="bg-surface-dim/50 p-3 rounded text-sm text-on-surface">
                     <span className="font-semibold text-white">{scoreMeaning}: </span>
                     The system is <strong>{confidence}% confident</strong> that the provided signature {isPositive ? 'is genuine and matches the reference profile.' : 'is a forgery and does not match the reference profile.'}
                  </div>
               </div>

               <div className="md:text-right shrink-0 bg-surface-dim/80 p-4 rounded-md border border-border">
                  <p className="font-semibold text-on-surface-muted text-xs uppercase mb-1">Sensitivity Limit (Threshold)</p>
                  <p className="text-white font-bold text-lg text-left">{result.threshold}</p>
                  <p className="text-[11px] text-on-surface-muted max-w-[150px] mt-2 leading-tight font-medium text-left">
                     Scores above this value are accepted. Lowering this makes the system stricter.
                  </p>
               </div>
            </div>
         </div>

         {/* Explanations */}
         {result.explanations && (
            <div className="glass-panel p-6">
               <div className="mb-6">
                  <h3 className="text-lg text-white font-semibold flex items-center gap-2">
                     <Sparkles className="text-primary" size={20} />
                     Forensic Visual Analysis
                  </h3>
                  <p className="text-sm text-on-surface-muted mt-1">
                     Visual breakdowns showing exactly which parts of the signature the AI focused on to make its decision.
                  </p>
               </div>

               {mode === 'baseline' ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
                     <ExplanationCard
                        title="Stroke Mapping (Overlay)"
                        description="Highlights structural differences mapped directly over the signature."
                        src={`${API_BASE}${result.explanations.overlay}`}
                     />
                     <ExplanationCard
                        title="Attention Heatmap (Saliency)"
                        description="Red zones show the exact curves and pressure points the AI flagged as important."
                        src={`${API_BASE}${result.explanations.heatmap}`}
                     />
                  </div>
               ) : (
                  <div className="space-y-8">
                     <div className="bg-surface-dim/30 p-4 rounded-md border border-border">
                        <h4 className="text-md text-primary mb-1 font-semibold">Reference Document (A)</h4>
                        <p className="text-xs text-on-surface-muted mb-4 font-medium">The AI analyzed this as the ground truth.</p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6">
                           <ExplanationCard
                              title="Stroke Mapping"
                              src={`${API_BASE}${result.explanations.overlay_a}`}
                           />
                           <ExplanationCard
                              title="Attention Heatmap"
                              src={`${API_BASE}${result.explanations.heatmap_a}`}
                           />
                        </div>
                     </div>

                     <div className="bg-surface-dim/30 p-4 rounded-md border border-border">
                        <h4 className="text-md text-[#ffb4ab] mb-1 font-semibold">Questioned Document (B)</h4>
                        <p className="text-xs text-on-surface-muted mb-4 font-medium">The AI compared this against the reference.</p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6">
                           <ExplanationCard
                              title="Stroke Mapping"
                              src={`${API_BASE}${result.explanations.overlay_b}`}
                           />
                           <ExplanationCard
                              title="Attention Heatmap"
                              src={`${API_BASE}${result.explanations.heatmap_b}`}
                           />
                        </div>
                     </div>
                  </div>
               )}
            </div>
         )}
      </div>
   )
}

function ExplanationCard({ title, description, src }) {
   return (
      <div className="flex flex-col h-full bg-surface-dim/50 rounded-md p-3 border border-border shadow-sm">
         <div className="mb-3">
            <p className="text-sm font-semibold text-white">{title}</p>
            {description && <p className="text-xs text-on-surface-muted mt-1 leading-relaxed font-medium">{description}</p>}
         </div>
         <div className="mt-auto bg-white rounded-sm flex items-center justify-center p-2 min-h-[120px] md:min-h-[160px] overflow-hidden border border-border/50">
            <img
               src={src}
               alt={title}
               className="max-w-full max-h-[180px] md:max-h-[250px] object-contain"
            />
         </div>
      </div>
   )
}
