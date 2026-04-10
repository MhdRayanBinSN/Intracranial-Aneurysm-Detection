import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import {
  CircleStackIcon,
  UsersIcon,
  BeakerIcon,
  ChartBarIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  MagnifyingGlassIcon,
  ServerStackIcon,
  CpuChipIcon,
} from '@heroicons/react/24/outline'
import api from '../api/client'

const FALLBACK = {
  overview: { total_series: 4348, positive_cases: 1864, negative_cases: 2484, positive_pct: 42.9, negative_pct: 57.1, total_locations: 13 },
  modality: { CTA: 1808, MRA: 1252, 'MRI T2': 983, 'MRI T1post': 305 },
  sex: { Female: 3005, Male: 1343 },
  age: { mean: 58.5, median: 59.0, min: 18, max: 89, buckets: [
    { label: '<30', count: 39 }, { label: '30-39', count: 217 }, { label: '40-49', count: 614 },
    { label: '50-59', count: 1165 }, { label: '60-69', count: 1321 }, { label: '70-79', count: 762 }, { label: '80+', count: 230 }
  ]},
  locations: [
    { name: 'Anterior Communicating Artery', count: 363, pct: 19.5 },
    { name: 'Left Supraclinoid Internal Carotid Artery', count: 331, pct: 17.8 },
    { name: 'Right Middle Cerebral Artery', count: 294, pct: 15.8 },
    { name: 'Right Supraclinoid Internal Carotid Artery', count: 277, pct: 14.9 },
    { name: 'Left Middle Cerebral Artery', count: 219, pct: 11.8 },
    { name: 'Other Posterior Circulation', count: 113, pct: 6.1 },
    { name: 'Basilar Tip', count: 110, pct: 5.9 },
    { name: 'Right Posterior Communicating Artery', count: 101, pct: 5.4 },
    { name: 'Right Infraclinoid Internal Carotid Artery', count: 98, pct: 5.3 },
    { name: 'Left Posterior Communicating Artery', count: 86, pct: 4.6 },
    { name: 'Left Infraclinoid Internal Carotid Artery', count: 78, pct: 4.2 },
    { name: 'Right Anterior Cerebral Artery', count: 56, pct: 3.0 },
    { name: 'Left Anterior Cerebral Artery', count: 46, pct: 2.5 },
  ],
  files: { zip_size_gb: 214.86 },
  competition: {
    name: 'RSNA 2023 Intracranial Aneurysm Detection',
    host: 'Radiological Society of North America (RSNA)',
    year: 2023, platform: 'Kaggle',
    task: 'Multi-label binary classification across 13 anatomical locations',
    imaging: 'CT Angiography (CTA), MR Angiography (MRA), MRI'
  }
}

function StatCard({ icon: Icon, label, value, sub, color = 'text-primary-400', delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="card p-5 hover:border-primary-500/30 transition-colors"
    >
      <div className={`w-10 h-10 rounded-lg ${color.replace('text-', 'bg-').replace('-400', '-500/15')} flex items-center justify-center mb-3`}>
        <Icon className={`w-5 h-5 ${color}`} />
      </div>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      <p className="text-sm font-medium text-surface-200 mt-1">{label}</p>
      {sub && <p className="text-xs text-surface-500 mt-0.5">{sub}</p>}
    </motion.div>
  )
}

function BarChart({ data, maxVal, colorClass }) {
  return (
    <div className="space-y-2.5">
      {data.map((item, i) => (
        <div key={i} className="group">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-surface-300 truncate max-w-[60%]">{item.name || item.label}</span>
            <span className="text-surface-400 font-medium">{item.count.toLocaleString()}{item.pct != null ? ` (${item.pct}%)` : ''}</span>
          </div>
          <div className="h-2 bg-surface-800 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(item.count / maxVal) * 100}%` }}
              transition={{ duration: 0.8, delay: i * 0.05, ease: 'easeOut' }}
              className={`h-full rounded-full ${colorClass}`}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

function DonutRing({ pct, label, color }) {
  const r = 36, circ = 2 * Math.PI * r
  const dash = (pct / 100) * circ
  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="96" height="96" viewBox="0 0 96 96">
        <circle cx="48" cy="48" r={r} fill="none" stroke="#1e293b" strokeWidth="10" />
        <motion.circle
          cx="48" cy="48" r={r}
          fill="none" stroke={color} strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={circ}
          initial={{ strokeDashoffset: circ }}
          animate={{ strokeDashoffset: circ - dash }}
          transition={{ duration: 1.2, ease: 'easeOut' }}
          style={{ transformOrigin: '48px 48px', transform: 'rotate(-90deg)' }}
        />
        <text x="48" y="53" textAnchor="middle" fontSize="14" fontWeight="bold" fill="white">{pct}%</text>
      </svg>
      <p className="text-xs text-surface-400 text-center">{label}</p>
    </div>
  )
}

export default function Dataset() {
  const [data, setData] = useState(FALLBACK)
  const [loading, setLoading] = useState(true)
  const [live, setLive] = useState(false)

  useEffect(() => {
    api.get('/dataset/info')
      .then(r => { if (r.data?.available) { setData(r.data); setLive(true) } })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const { overview, modality, sex, age, locations, files, competition } = data
  const maxLocCount = Math.max(...locations.map(l => l.count))
  const maxBucket = Math.max(...age.buckets.map(b => b.count))
  const modList = Object.entries(modality).sort((a,b) => b[1]-a[1])
  const maxMod = modList[0]?.[1] || 1

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-7xl mx-auto space-y-8">

        {/* ── Header ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-primary-500 flex items-center justify-center">
              <CircleStackIcon className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-4xl font-bold">
              Dataset <span className="text-primary-400">Information</span>
            </h1>
          </div>
          <p className="text-surface-400 max-w-2xl mx-auto text-sm">
            Comprehensive statistics for the <strong className="text-white">RSNA 2023 Intracranial Aneurysm Detection</strong> dataset
            used to train and evaluate our detection models.
          </p>
          <div className="flex items-center justify-center gap-2 mt-3">
            <div className={`w-2 h-2 rounded-full ${live ? 'bg-green-400' : 'bg-amber-400'} animate-pulse`} />
            <span className="text-xs text-surface-500">{live ? 'Live data from train.csv' : 'Using cached statistics'}</span>
          </div>
        </motion.div>

        {/* ── Competition Info ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
          className="card p-6 bg-gradient-to-r from-primary-500/10 to-blue-500/10 border-primary-500/20"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: 'Competition', value: competition.name, icon: '🏆' },
              { label: 'Host', value: competition.host, icon: '🏥' },
              { label: 'Platform', value: `${competition.platform} · ${competition.year}`, icon: '💻' },
              { label: 'Task', value: competition.task, icon: '🎯' },
            ].map((item, i) => (
              <div key={i} className="space-y-1">
                <p className="text-xs text-surface-500 uppercase tracking-wider">{item.icon} {item.label}</p>
                <p className="text-sm text-surface-200 font-medium leading-snug">{item.value}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* ── Key Stats ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard icon={CircleStackIcon} label="Total Series" value={overview.total_series.toLocaleString()} sub="CT/MRI scan series" color="text-primary-400" delay={0.1} />
          <StatCard icon={ExclamationTriangleIcon} label="Aneurysm Positive" value={overview.positive_cases.toLocaleString()} sub={`${overview.positive_pct}% of dataset`} color="text-risk-high" delay={0.15} />
          <StatCard icon={CheckCircleIcon} label="Aneurysm Negative" value={overview.negative_cases.toLocaleString()} sub={`${overview.negative_pct}% of dataset`} color="text-risk-low" delay={0.2} />
          <StatCard icon={ServerStackIcon} label="Dataset Size" value={`${files.zip_size_gb} GB`} sub="Raw ZIP archive" color="text-blue-400" delay={0.25} />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard icon={UsersIcon} label="Mean Patient Age" value={age.mean} sub={`Range: ${age.min}–${age.max} yrs`} color="text-amber-400" delay={0.1} />
          <StatCard icon={BeakerIcon} label="Scan Modalities" value={Object.keys(modality).length} sub="CTA, MRA, MRI T1/T2" color="text-violet-400" delay={0.15} />
          <StatCard icon={MagnifyingGlassIcon} label="Anatomical Locations" value={overview.total_locations} sub="RSNA standard zones" color="text-cyan-400" delay={0.2} />
          <StatCard icon={CpuChipIcon} label="Classification Labels" value={`${overview.total_locations + 1}`} sub="13 locations + overall" color="text-pink-400" delay={0.25} />
        </div>

        {/* ── Middle Row: Donut charts + Modality ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Class Balance */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="card p-6">
            <h2 className="font-semibold text-lg mb-6 flex items-center gap-2">
              <ChartBarIcon className="w-5 h-5 text-primary-400" /> Class Balance
            </h2>
            <div className="flex justify-around">
              <DonutRing pct={overview.positive_pct} label={`Positive\n(${overview.positive_cases.toLocaleString()})`} color="#ef4444" />
              <DonutRing pct={overview.negative_pct} label={`Negative\n(${overview.negative_cases.toLocaleString()})`} color="#22c55e" />
            </div>
            <p className="text-xs text-surface-500 text-center mt-4">
              ⚠️ Moderately imbalanced — focal loss applied during training
            </p>
          </motion.div>

          {/* Sex Distribution */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.32 }} className="card p-6">
            <h2 className="font-semibold text-lg mb-6 flex items-center gap-2">
              <UsersIcon className="w-5 h-5 text-violet-400" /> Sex Distribution
            </h2>
            <div className="flex justify-around">
              {Object.entries(sex).map(([s, count], i) => (
                <DonutRing
                  key={s} pct={Math.round(count / overview.total_series * 100)}
                  label={`${s}\n(${count.toLocaleString()})`}
                  color={s === 'Female' ? '#a78bfa' : '#60a5fa'}
                />
              ))}
            </div>
            <p className="text-xs text-surface-500 text-center mt-4">
              Female-predominant cohort (consistent with IA literature)
            </p>
          </motion.div>

          {/* Modality breakdown */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.34 }} className="card p-6">
            <h2 className="font-semibold text-lg mb-6 flex items-center gap-2">
              <BeakerIcon className="w-5 h-5 text-cyan-400" /> Imaging Modalities
            </h2>
            <BarChart
              data={modList.map(([name, count]) => ({ name, count, pct: Math.round(count / overview.total_series * 100) }))}
              maxVal={maxMod}
              colorClass="bg-gradient-to-r from-cyan-500 to-primary-500"
            />
          </motion.div>
        </div>

        {/* ── Age Distribution ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.36 }} className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="font-semibold text-lg flex items-center gap-2">
              <ClockIcon className="w-5 h-5 text-amber-400" /> Age Distribution
            </h2>
            <div className="flex gap-4 text-sm">
              <span className="text-surface-400">Mean: <strong className="text-white">{age.mean} yrs</strong></span>
              <span className="text-surface-400">Median: <strong className="text-white">{age.median} yrs</strong></span>
              <span className="text-surface-400">Range: <strong className="text-white">{age.min}–{age.max}</strong></span>
            </div>
          </div>
          <div className="flex items-end gap-3 h-40">
            {age.buckets.map((b, i) => {
              const heightPct = Math.round((b.count / maxBucket) * 100)
              return (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  <p className="text-xs text-surface-500">{b.count.toLocaleString()}</p>
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height: `${heightPct}%` }}
                    transition={{ duration: 0.8, delay: i * 0.07, ease: 'easeOut' }}
                    className="w-full rounded-t-md bg-gradient-to-t from-amber-600 to-amber-400 min-h-[4px]"
                    style={{ maxHeight: '120px' }}
                  />
                  <p className="text-xs text-surface-400">{b.label}</p>
                </div>
              )
            })}
          </div>
        </motion.div>

        {/* ── Aneurysm Location Distribution ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="card p-6">
          <h2 className="font-semibold text-xl mb-2 flex items-center gap-2">
            <MagnifyingGlassIcon className="w-6 h-6 text-risk-high" /> Aneurysm Location Distribution
          </h2>
          <p className="text-surface-500 text-sm mb-6">
            Number of positive cases per anatomical location (among {overview.positive_cases.toLocaleString()} positive series).
            One series can have multiple locations.
          </p>
          <BarChart
            data={locations}
            maxVal={maxLocCount}
            colorClass="bg-gradient-to-r from-risk-high/80 to-risk-high"
          />
        </motion.div>

        {/* ── Dataset Files ── */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.45 }} className="card p-6">
          <h2 className="font-semibold text-lg mb-4 flex items-center gap-2">
            <ServerStackIcon className="w-5 h-5 text-blue-400" /> Dataset Files
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { label: 'Main Archive', path: 'rsna-intracranial-aneurysm-detection.zip', size: `${files.zip_size_gb} GB`, color: 'text-blue-400' },
              { label: 'Labels CSV', path: 'train.csv', size: '4,348 rows × 18 cols', color: 'text-green-400' },
              { label: 'Location', path: 'C:\\Users\\Rayan\\Desktop\\Main Project\\', size: 'Local storage', color: 'text-surface-400' },
            ].map((f, i) => (
              <div key={i} className="bg-surface-800/50 rounded-xl p-4 border border-surface-700">
                <p className="text-xs text-surface-500 uppercase tracking-wider mb-1">{f.label}</p>
                <p className={`text-sm font-mono break-all ${f.color}`}>{f.path}</p>
                <p className="text-xs text-surface-500 mt-1">{f.size}</p>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Disclaimer */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
          className="p-4 bg-surface-900/50 border border-surface-800 rounded-xl"
        >
          <p className="text-xs text-surface-500 text-center">
            Data sourced from the <strong className="text-surface-400">RSNA 2023 Intracranial Aneurysm Detection</strong> Kaggle competition.
            Used solely for academic and research purposes. All patient data is de-identified in accordance with competition rules.
          </p>
        </motion.div>
      </div>
    </div>
  )
}
