"use client";

import React, { useEffect, useState } from "react";
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, LineChart, Line, AreaChart, Area, CartesianGrid 
} from "recharts";
import { motion } from "framer-motion";
import { Brain, Activity, Target, Zap, AlertTriangle, Image as ImageIcon, UploadCloud, Key, Settings } from "lucide-react";

export default function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [grokKey, setGrokKey] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  const fetchData = () => {
    fetch("/api/data")
      .then((res) => {
        if (!res.ok) throw new Error("Data fetch failed. API Route returned an error.");
        return res.json();
      })
      .then((d) => {
        if (d.error) setError(d.error);
        else setData(d);
      })
      .catch((e) => {
        console.error("Error loading results:", e);
        setError(e.message || "Failed to parse API data.");
      });
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleAnalyze = async () => {
    if (!file) {
      alert("Please upload a .conllu dataset first.");
      return;
    }
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("grokKey", grokKey);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const respData = await res.json();
      if (!res.ok) throw new Error(respData.error);
      
      // Reload Data
      fetchData();
      alert("Analysis complete! Graphs updated.");
    } catch (e) {
      alert("Analysis failed: " + e.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (error) {
    return (
      <div className="flex items-center justify-center h-screen w-full bg-slate-900 text-rose-400">
        <div className="bg-white/10 p-8 rounded-xl border border-rose-500/30 text-center">
          <AlertTriangle size={64} className="mx-auto mb-4 text-rose-500" />
          <h2 className="text-2xl font-bold mb-2">Error Loading Data</h2>
          <p>{error}</p>
          <button onClick={() => window.location.reload()} className="mt-6 px-4 py-2 bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 rounded font-bold transition-colors">Retry</button>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-screen w-full bg-slate-900 text-slate-200">
        <motion.div 
          animate={{ scale: [1, 1.2, 1], rotate: 360 }} 
          transition={{ repeat: Infinity, duration: 2 }}
          className="flex flex-col items-center gap-4"
        >
          <Brain size={64} className="text-blue-500" />
          <h2 className="text-xl font-bold animate-pulse">Initializing AI Models...</h2>
        </motion.div>
      </div>
    );
  }

  const taggers = Object.keys(data.taggers || {});
  
  // Format accuracy for BarChart
  const accuracyData = taggers.map(tagger => ({
    name: tagger.split(" ")[0],
    fullName: tagger,
    accuracy: (data.taggers[tagger].accuracy * 100).toFixed(2),
    error: (data.taggers[tagger].error_rate * 100).toFixed(2)
  }));

  const containerVars = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };
  
  const itemVars = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="min-h-screen p-6 md:p-12 pb-24 font-sans max-w-7xl mx-auto">
      
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-12 flex flex-col md:flex-row justify-between items-start md:items-end gap-6"
      >
        <div>
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-blue-500/20 p-2 rounded-lg">
              <Brain className="text-blue-400" size={32} />
            </div>
            <h1 className="text-4xl font-extrabold text-white">POS Analysis Dashboard</h1>
          </div>
          <p className="text-slate-400 text-lg">Comprehensive error evaluation of {data.meta.dataset}</p>
        </div>
        <div className="glass-card !py-3 !px-6 flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-slate-400 uppercase tracking-wider font-semibold">Test Tokens</p>
            <p className="text-2xl font-bold text-white">{data.meta.test_tokens.toLocaleString()}</p>
          </div>
          <div className="h-10 w-px bg-white/20"></div>
          <div className="text-right">
            <p className="text-sm text-slate-400 uppercase tracking-wider font-semibold">Sentences</p>
            <p className="text-2xl font-bold text-white">{data.meta.test_sentences.toLocaleString()}</p>
          </div>
        </div>
      </motion.header>

      {/* NEW UPLOAD SECTION */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card mb-12 flex flex-col md:flex-row gap-6 items-center p-6 bg-indigo-900/20 border-indigo-500/30">
        <div className="flex-1 w-full">
          <label className="block text-sm font-medium text-slate-300 mb-2">Upload Dataset (.conllu)</label>
          <div className="w-full relative">
            <input type="file" accept=".conllu,.txt" onChange={(e) => setFile(e.target.files[0])} className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" />
            <div className={`p-4 border-2 border-dashed rounded-xl flex items-center justify-center gap-3 transition-colors ${file ? 'border-emerald-500/50 bg-emerald-500/10' : 'border-indigo-500/30 bg-indigo-500/5 hover:border-indigo-500/60'}`}>
              <UploadCloud className={file ? "text-emerald-400" : "text-indigo-400"} />
              <span className="text-slate-300 font-medium truncate max-w-[200px]">{file ? file.name : "Drag & Drop or Browser"}</span>
            </div>
          </div>
        </div>
        {showAdvanced && (
          <div className="flex-1 w-full relative">
            <label className="block text-sm font-medium text-slate-300 mb-2">Grok API Key (Optional)</label>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
              <input 
                type="password" 
                placeholder="xAI Key (xai-...)" 
                value={grokKey} 
                onChange={(e) => setGrokKey(e.target.value)} 
                className="w-full bg-slate-900/50 border border-slate-700 rounded-xl py-3 pl-10 pr-4 text-slate-200 outline-none focus:border-indigo-500 transition-colors placeholder:text-slate-600"
              />
            </div>
          </div>
        )}
        <div className="flex-none self-end pb-0.5 w-full md:w-auto flex flex-col items-center">
          <button 
            onClick={handleAnalyze} 
            disabled={isAnalyzing || !file}
            className="w-full md:w-auto px-8 py-3 h-[50px] bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-xl font-bold transition-colors flex items-center justify-center gap-2"
          >
            {isAnalyzing ? (
              <><Zap size={18} className="animate-spin text-amber-400" /> Analyzing...</>
            ) : "Analyze Dataset"}
          </button>
          <button 
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-xs text-slate-400 mt-3 hover:text-indigo-400 flex items-center gap-1 transition-colors"
          >
            <Settings size={12} /> {showAdvanced ? "Hide Advanced" : "Advanced Settings"}
          </button>
        </div>
      </motion.div>

      <motion.div variants={containerVars} initial="hidden" animate="show" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        {accuracyData.map((t, idx) => (
          <motion.div key={t.name} variants={itemVars} className="glass-card relative overflow-hidden group">
            <div className={`absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-blue-500/10 to-transparent rounded-bl-full -z-10 transition-transform group-hover:scale-110`} />
            <h3 className="text-xl font-bold text-slate-100 flex items-center gap-2">
              <Target size={20} className="text-indigo-400" />
              {t.name}
            </h3>
            <p className="text-sm text-slate-400 truncate mb-4" title={t.fullName}>{t.fullName}</p>
            <div className="flex justify-between items-end">
              <div>
                <p className="text-4xl font-extrabold text-white">{t.accuracy}<span className="text-lg text-slate-500">%</span></p>
                <p className="text-sm text-emerald-400 font-medium mt-1">Accuracy</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-slate-300">{t.error}<span className="text-sm text-slate-500">%</span></p>
                <p className="text-sm text-rose-400 font-medium mt-1">Error Rate</p>
              </div>
            </div>
            
            {/* simple bar */}
            <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4 overflow-hidden">
              <motion.div 
                initial={{ width: 0 }}
                animate={{ width: `${t.accuracy}%` }}
                transition={{ duration: 1, delay: 0.2 + (idx * 0.1) }}
                className="bg-indigo-500 h-1.5 rounded-full"
              />
            </div>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }} className="glass-card lg:col-span-2">
          <div className="flex items-center gap-2 mb-6">
            <Activity className="text-indigo-400" />
            <h2 className="text-2xl font-bold text-white">Overall Accuracy Comparison</h2>
          </div>
          <div className="h-[300px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={accuracyData} layout="vertical" margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#334155" />
                <XAxis type="number" domain={[0, 100]} stroke="#94a3b8" />
                <YAxis dataKey="name" type="category" stroke="#f8fafc" fontWeight="bold" width={80} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(15, 23, 42, 0.9)', borderColor: '#334155', borderRadius: '12px', color: '#fff' }}
                  itemStyle={{ color: '#fff' }}
                  cursor={{fill: 'rgba(255,255,255,0.05)'}}
                />
                <Bar dataKey="accuracy" fill="#4f46e5" radius={[0, 6, 6, 0]} barSize={32} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }} className="glass-card">
          <div className="flex items-center gap-2 mb-6">
            <Zap className="text-amber-400" />
            <h2 className="text-2xl font-bold text-white">Project Innovations</h2>
          </div>
          <ul className="space-y-4">
            {[
              "Majority-Vote Ensemble Tagger",
              "Taxonomy defined for Cross-Tagger errors",
              "Hard-Case Extraction algorithms",
              "Zero-Shot visualization insights"
            ].map((innov, i) => (
              <li key={i} className="flex items-start gap-3 p-3 rounded-lg bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
                <span className="flex-shrink-0 w-6 h-6 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-sm font-bold mt-0.5">{i+1}</span>
                <span className="text-slate-200">{innov}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>

      <div className="mb-8 flex items-center gap-2">
        <ImageIcon className="text-pink-400 flex-shrink-0" />
        <h2 className="text-3xl font-extrabold text-white tracking-tight">Visual Analysis Generated</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          { file: "error_taxonomy.png", title: "Overall Error Taxonomy", colSpan: 1 },
          { file: "overall_accuracy.png", title: "Accuracy Overview", colSpan: 1 },
          { file: "ensemble_comparison.png", title: "Ensemble Performance Gain", colSpan: 1 },
          { file: "per_tag_f1_comparison.png", title: "Per-Tag F1 Benchmark", colSpan: 2 },
          { file: "position_error_rates.png", title: "Error Rate by Sentence Position", colSpan: 1 },
          { file: "oov_analysis.png", title: "OOV vs In-Vocabulary Impacts", colSpan: 1 },
          { file: "frequency_bucket_errors.png", title: "Error Rates by Word Frequency", colSpan: 2 },
          ...Object.keys(data.individual_taggers || {}).map(name => ({
            file: `confusion_${name.replace(/ /g, '_').replace(/\(/g, '').replace(/\)/g, '')}.png`,
            title: `Confusion Matrix (${name})`,
            colSpan: 1
          }))
        ].map((img, i) => (
          <motion.div 
            key={img.file}
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: i * 0.1 }}
            className={`glass-card p-4 flex flex-col items-center justify-center group cursor-pointer ${img.colSpan === 2 ? 'md:col-span-2' : img.colSpan === 3 ? 'md:col-span-2 lg:col-span-3' : ''}`}
          >
            <h3 className="text-lg font-semibold text-slate-300 mb-4 self-start w-full border-b border-white/10 pb-2">{img.title}</h3>
            <div className="relative w-full rounded-lg overflow-hidden bg-white/5 flex items-center justify-center min-h-[250px]">
              {img.placeholder ? (
                <div className="text-center p-8">
                  <AlertTriangle size={48} className="text-amber-500 mx-auto mb-4 opacity-70" />
                  <p className="text-slate-400">Please refer to individual tagger confusion matrices in the results folder.</p>
                </div>
              ) : (
                  <img 
                    src={`/api/images/${img.file}`} 
                    alt={img.title}
                    onClick={() => setSelectedImage(`/api/images/${img.file}`)}
                    className="w-full h-auto object-contain max-h-[400px] cursor-pointer transition-transform duration-500 group-hover:scale-[1.02]"
                    onError={(e) => {
                      e.target.style.display = 'none';
                      e.target.nextSibling.style.display = 'flex';
                    }}
                  />
              )}
              {!img.placeholder && (
                <div className="hidden absolute inset-0 items-center justify-center bg-slate-900/50 flex-col">
                  <AlertTriangle className="text-slate-500 mb-2" />
                  <span className="text-slate-500 text-sm">Image unavailable</span>
                </div>
              )}
            </div>
          </motion.div>
        ))}
      </div>
      
      {/* LIGHTBOX MODAL */}
      {selectedImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 cursor-zoom-out"
          onClick={() => setSelectedImage(null)}
        >
          <img 
            src={selectedImage} 
            className="max-w-full max-h-[90vh] object-contain rounded-xl shadow-2xl" 
            alt="Expanded preview"
          />
        </div>
      )}
    </div>
  );
}
