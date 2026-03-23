import { useState, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Treemap, Legend } from "recharts";

const RAW_DATA = [
  {id:49,count:60751,label:"Adverse effects risk increase"},
  {id:47,count:34360,label:"Metabolism decrease"},
  {id:73,count:23779,label:"Serum concentration increase"},
  {id:75,count:9470,label:"Serum concentration decrease"},
  {id:60,count:8397,label:"Hypotensive activities increase"},
  {id:70,count:7786,label:"Therapeutic efficacy decrease"},
  {id:20,count:6140,label:"QTc-prolonging activities"},
  {id:16,count:5413,label:"CNS depressant activities"},
  {id:4,count:5011,label:"Metabolism increase"},
  {id:6,count:3160,label:"Anticoagulant activities increase"},
  {id:37,count:3089,label:"Antihypertensive activities decrease"},
  {id:9,count:2109,label:"Hypoglycemic activities"},
  {id:72,count:1825,label:"Excretion rate decrease"},
  {id:54,count:1277,label:"Bradycardic activities"},
  {id:83,count:1204,label:"Hypokalemic activities"},
  {id:58,count:1043,label:"Cardiotoxic activities decrease"},
  {id:32,count:1011,label:"Sedative activities increase"},
  {id:27,count:936,label:"Neuroexcitatory activities"},
  {id:67,count:930,label:"Absorption decrease"},
  {id:64,count:803,label:"Serotonergic activities"},
  {id:25,count:716,label:"AV block activities"},
  {id:71,count:673,label:"Hypertensive activities"},
  {id:57,count:664,label:"Nephrotoxic activities"},
  {id:10,count:629,label:"Antihypertensive activities increase"},
  {id:30,count:616,label:"Orthostatic hypotensive activities"},
  {id:76,count:558,label:"Sedative activities decrease"},
  {id:77,count:538,label:"Active metabolites increase"},
  {id:3,count:518,label:"Bioavailability decrease"},
  {id:61,count:498,label:"Stimulatory activities decrease"},
  {id:33,count:442,label:"QTc prolongation risk"},
  {id:21,count:428,label:"Neuromuscular blocking increase"},
  {id:74,count:422,label:"Fluid retaining activities"},
  {id:85,count:372,label:"Tachycardic activities"},
  {id:14,count:361,label:"Bronchodilatory activities decrease"},
  {id:82,count:355,label:"Arrhythmogenic activities"},
  {id:53,count:336,label:"Antiplatelet activities increase"},
  {id:29,count:324,label:"Diuretic activities decrease"},
  {id:2,count:323,label:"Anticholinergic activities"},
  {id:34,count:312,label:"Immunosuppressive activities"},
  {id:11,count:312,label:"Active metabolites decrease"},
  {id:5,count:309,label:"Vasoconstricting activities decrease"},
  {id:40,count:300,label:"Respiratory depressant activities"},
  {id:69,count:280,label:"Analgesic activities increase"},
  {id:68,count:278,label:"Hyperkalemic activities"},
  {id:8,count:244,label:"Therapeutic efficacy increase"},
  {id:12,count:238,label:"Anticoagulant activities decrease"},
  {id:15,count:202,label:"Cardiotoxic activities increase"},
  {id:24,count:180,label:"Hypocalcemic activities"},
  {id:39,count:148,label:"Constipating activities"},
  {id:66,count:128,label:"Bleeding risk increase"},
  {id:55,count:118,label:"Hyponatremic activities"},
  {id:19,count:109,label:"Vasoconstricting activities increase"},
  {id:81,count:108,label:"Thrombogenic activities"},
  {id:36,count:94,label:"Antipsychotic activities"},
  {id:22,count:94,label:"Adverse neuromuscular activities"},
  {id:51,count:83,label:"Hypercalcemic activities"},
  {id:17,count:83,label:"Neuromuscular blocking decrease"},
  {id:18,count:82,label:"Absorption increase"},
  {id:48,count:69,label:"Rhabdomyolysis activities"},
  {id:35,count:69,label:"Neurotoxic activities"},
  {id:84,count:65,label:"Vasopressor activities"},
  {id:80,count:64,label:"Hepatotoxic activities"},
  {id:23,count:56,label:"Stimulatory activities increase"},
  {id:13,count:45,label:"Absorption decrease (combined)"},
  {id:59,count:43,label:"Ulcerogenic activities"},
  {id:63,count:34,label:"Myelosuppressive activities"},
  {id:56,count:33,label:"Hypotension risk increase"},
  {id:45,count:33,label:"Diagnostic agent decrease"},
  {id:38,count:33,label:"Vasodilatory activities"},
  {id:65,count:32,label:"Excretion rate increase"},
  {id:78,count:28,label:"Hyperglycemic activities"},
  {id:86,count:27,label:"Hypersensitivity reaction risk"},
  {id:79,count:27,label:"CNS depressant + hypertension"},
  {id:50,count:26,label:"Heart failure risk increase"},
  {id:46,count:26,label:"Bronchoconstrictory activities"},
  {id:7,count:21,label:"Ototoxic activities"},
  {id:41,count:14,label:"Hypotensive + CNS depressant"},
  {id:31,count:14,label:"Hypertension risk increase"},
  {id:44,count:13,label:"Central neurotoxic activities"},
  {id:62,count:11,label:"Bioavailability increase"},
  {id:43,count:11,label:"Protein binding decrease"},
  {id:28,count:11,label:"Dermatologic adverse activities"},
  {id:1,count:11,label:"Photosensitizing activities"},
  {id:52,count:10,label:"Analgesic activities decrease"},
  {id:26,count:7,label:"Antiplatelet activities decrease"},
  {id:42,count:6,label:"Hyperkalemia risk increase"},
];

const TOTAL = RAW_DATA.reduce((s, d) => s + d.count, 0);

const COLORS = [
  "#6366f1","#8b5cf6","#a78bfa","#c4b5fd","#818cf8",
  "#60a5fa","#38bdf8","#22d3ee","#2dd4bf","#34d399",
  "#4ade80","#a3e635","#facc15","#fb923c","#f87171",
  "#e879f9","#f472b6","#fbbf24","#a8a29e","#94a3b8",
];

function getColor(i) {
  return COLORS[i % COLORS.length];
}

const ImbalanceRatio = ({ data }) => {
  const maxCount = data[0].count;
  const minCount = data[data.length - 1].count;
  const ratio = (maxCount / minCount).toFixed(0);
  const median = data[Math.floor(data.length / 2)].count;
  const mean = Math.round(TOTAL / data.length);

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 mb-6">
      {[
        { label: "Total Samples", value: TOTAL.toLocaleString(), color: "bg-indigo-50 text-indigo-700 border-indigo-200" },
        { label: "Num Classes", value: data.length, color: "bg-purple-50 text-purple-700 border-purple-200" },
        { label: "Imbalance Ratio", value: `${ratio}:1`, color: "bg-red-50 text-red-700 border-red-200" },
        { label: "Max / Min", value: `${maxCount.toLocaleString()} / ${minCount}`, color: "bg-amber-50 text-amber-700 border-amber-200" },
      ].map((s) => (
        <div key={s.label} className={`rounded-xl border p-4 ${s.color}`}>
          <div className="text-xs font-medium opacity-70 uppercase tracking-wide">{s.label}</div>
          <div className="text-2xl font-bold mt-1">{s.value}</div>
        </div>
      ))}
    </div>
  );
};

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-sm max-w-xs">
      <div className="font-semibold text-gray-800">Class {d.id}: {d.label}</div>
      <div className="text-gray-600 mt-1">Count: <span className="font-bold text-indigo-600">{d.count.toLocaleString()}</span></div>
      <div className="text-gray-500">Share: {((d.count / TOTAL) * 100).toFixed(2)}%</div>
    </div>
  );
};

const TreemapContent = ({ x, y, width, height, name, count, index }) => {
  if (width < 30 || height < 20) return null;
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} fill={getColor(index)} stroke="#fff" strokeWidth={2} rx={4} />
      {width > 50 && height > 35 && (
        <>
          <text x={x + width / 2} y={y + height / 2 - 6} textAnchor="middle" fill="#fff" fontSize={11} fontWeight="bold">{name}</text>
          <text x={x + width / 2} y={y + height / 2 + 10} textAnchor="middle" fill="rgba(255,255,255,0.8)" fontSize={9}>{count?.toLocaleString()}</text>
        </>
      )}
    </g>
  );
};

export default function Dashboard() {
  const [view, setView] = useState("bar");
  const [topN, setTopN] = useState(20);
  const [logScale, setLogScale] = useState(false);

  const sortedData = useMemo(() => [...RAW_DATA].sort((a, b) => b.count - a.count), []);
  const displayData = useMemo(() => sortedData.slice(0, topN), [sortedData, topN]);

  const pieData = useMemo(() => {
    const top10 = sortedData.slice(0, 10);
    const otherCount = TOTAL - top10.reduce((s, d) => s + d.count, 0);
    return [...top10.map((d) => ({ name: `Class ${d.id}`, value: d.count })), { name: "Other (76 classes)", value: otherCount }];
  }, [sortedData]);

  const treemapData = useMemo(
    () => sortedData.map((d) => ({ name: `C${d.id}`, count: d.count, size: d.count })),
    [sortedData]
  );

  const cumulativeData = useMemo(() => {
    let cum = 0;
    return sortedData.map((d, i) => {
      cum += d.count;
      return { ...d, cumPct: +((cum / TOTAL) * 100).toFixed(2), rank: i + 1 };
    });
  }, [sortedData]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-indigo-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800">DrugBank DDI — Class Imbalance Dashboard</h1>
          <p className="text-gray-500 mt-1">Visualizing {TOTAL.toLocaleString()} drug-drug interactions across {RAW_DATA.length} interaction types</p>
        </div>

        <ImbalanceRatio data={sortedData} />

        {/* Navigation */}
        <div className="flex flex-wrap gap-2 mb-6">
          {[
            { key: "bar", label: "Bar Chart" },
            { key: "log", label: "Log Scale Bar" },
            { key: "pie", label: "Pie Chart" },
            { key: "treemap", label: "Treemap" },
            { key: "cumulative", label: "Cumulative %" },
            { key: "table", label: "Data Table" },
          ].map((t) => (
            <button
              key={t.key}
              onClick={() => setView(t.key)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                view === t.key
                  ? "bg-indigo-600 text-white shadow-md"
                  : "bg-white text-gray-600 border border-gray-200 hover:bg-indigo-50 hover:text-indigo-600"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Top N slider for bar charts */}
        {(view === "bar" || view === "log") && (
          <div className="flex items-center gap-3 mb-4 bg-white rounded-lg border border-gray-200 p-3 w-fit">
            <span className="text-sm text-gray-600 font-medium">Show top</span>
            <input
              type="range" min={10} max={86} value={topN}
              onChange={(e) => setTopN(+e.target.value)}
              className="w-48 accent-indigo-600"
            />
            <span className="text-sm font-bold text-indigo-600 w-8">{topN}</span>
            <span className="text-sm text-gray-500">classes</span>
          </div>
        )}

        {/* Charts */}
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-6">
          {(view === "bar" || view === "log") && (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">
                {view === "log" ? "Log-Scale" : ""} Class Distribution (Top {topN})
              </h2>
              <ResponsiveContainer width="100%" height={500}>
                <BarChart data={displayData} margin={{ top: 5, right: 20, bottom: 60, left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="id"
                    tick={{ fontSize: 10 }}
                    angle={-45}
                    textAnchor="end"
                    label={{ value: "Class ID", position: "insideBottom", offset: -50, style: { fontSize: 12, fill: "#666" } }}
                  />
                  <YAxis
                    scale={view === "log" ? "log" : "auto"}
                    domain={view === "log" ? [1, "auto"] : [0, "auto"]}
                    tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v}
                    label={{ value: "Count", angle: -90, position: "insideLeft", offset: -45, style: { fontSize: 12, fill: "#666" } }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {displayData.map((_, i) => (
                      <Cell key={i} fill={getColor(i)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {view === "pie" && (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">Top 10 Classes vs Rest</h2>
              <ResponsiveContainer width="100%" height={500}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%" cy="50%"
                    outerRadius={200}
                    innerRadius={80}
                    dataKey="value"
                    label={({ name, percent }) => `${name} (${(percent * 100).toFixed(1)}%)`}
                    labelLine={{ strokeWidth: 1 }}
                  >
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={i === pieData.length - 1 ? "#d1d5db" : getColor(i)} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => v.toLocaleString()} />
                </PieChart>
              </ResponsiveContainer>
              <p className="text-center text-sm text-gray-500 mt-2">
                The top 10 classes account for <span className="font-bold text-indigo-600">
                  {((pieData.slice(0, 10).reduce((s, d) => s + d.value, 0) / TOTAL) * 100).toFixed(1)}%
                </span> of all interactions
              </p>
            </div>
          )}

          {view === "treemap" && (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">Treemap — Area Proportional to Class Size</h2>
              <ResponsiveContainer width="100%" height={500}>
                <Treemap
                  data={treemapData}
                  dataKey="size"
                  nameKey="name"
                  content={<TreemapContent />}
                />
              </ResponsiveContainer>
            </div>
          )}

          {view === "cumulative" && (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">Cumulative Distribution (Pareto)</h2>
              <ResponsiveContainer width="100%" height={500}>
                <BarChart data={cumulativeData} margin={{ top: 5, right: 20, bottom: 60, left: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="rank"
                    tick={{ fontSize: 10 }}
                    label={{ value: "Class Rank (by size)", position: "insideBottom", offset: -50, style: { fontSize: 12, fill: "#666" } }}
                  />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    domain={[0, 100]}
                    label={{ value: "Cumulative %", angle: -90, position: "insideLeft", offset: -45, style: { fontSize: 12, fill: "#666" } }}
                  />
                  <Tooltip
                    formatter={(v, name, props) => {
                      const d = props.payload;
                      return [`${v}% (Class ${d.id}: ${d.count.toLocaleString()})`, "Cumulative"];
                    }}
                  />
                  <Bar dataKey="cumPct" radius={[2, 2, 0, 0]}>
                    {cumulativeData.map((d, i) => (
                      <Cell key={i} fill={d.cumPct <= 80 ? "#6366f1" : d.cumPct <= 95 ? "#f59e0b" : "#ef4444"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="flex justify-center gap-6 mt-3 text-sm">
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-indigo-500 inline-block" /> 0–80%</span>
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-amber-500 inline-block" /> 80–95%</span>
                <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500 inline-block" /> 95–100%</span>
              </div>
            </div>
          )}

          {view === "table" && (
            <div>
              <h2 className="text-lg font-semibold text-gray-700 mb-4">Full Class Distribution Table</h2>
              <div className="overflow-auto max-h-96 rounded-lg border border-gray-100">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      <th className="text-left p-3 font-semibold text-gray-600">Rank</th>
                      <th className="text-left p-3 font-semibold text-gray-600">Class</th>
                      <th className="text-left p-3 font-semibold text-gray-600">Interaction Type</th>
                      <th className="text-right p-3 font-semibold text-gray-600">Count</th>
                      <th className="text-right p-3 font-semibold text-gray-600">Share %</th>
                      <th className="text-left p-3 font-semibold text-gray-600">Distribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedData.map((d, i) => (
                      <tr key={d.id} className={i % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                        <td className="p-3 text-gray-500">{i + 1}</td>
                        <td className="p-3 font-medium text-gray-800">{d.id}</td>
                        <td className="p-3 text-gray-600 max-w-xs truncate">{d.label}</td>
                        <td className="p-3 text-right font-mono text-gray-800">{d.count.toLocaleString()}</td>
                        <td className="p-3 text-right font-mono text-gray-600">{((d.count / TOTAL) * 100).toFixed(2)}%</td>
                        <td className="p-3">
                          <div className="w-32 bg-gray-200 rounded-full h-2">
                            <div
                              className="h-2 rounded-full"
                              style={{
                                width: `${Math.max(1, (d.count / sortedData[0].count) * 100)}%`,
                                backgroundColor: getColor(i),
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Insight box */}
        <div className="mt-6 bg-amber-50 border border-amber-200 rounded-xl p-5">
          <h3 className="font-semibold text-amber-800 mb-2">Key Imbalance Insights</h3>
          <div className="text-sm text-amber-700 space-y-1">
            <p>The top 3 classes (49, 47, 73) alone hold <strong>{((( 60751+34360+23779) / TOTAL) * 100).toFixed(1)}%</strong> of all data — a severe long-tail distribution.</p>
            <p>The imbalance ratio is <strong>10,125:1</strong> (Class 49 with 60,751 vs Class 42 with just 6 samples).</p>
            <p>Consider techniques like SMOTE, class weighting, focal loss, or hierarchical classification to handle this imbalance.</p>
          </div>
        </div>

        <p className="text-center text-xs text-gray-400 mt-6">DrugBank Drug-Drug Interaction Dataset — {TOTAL.toLocaleString()} interactions · {RAW_DATA.length} classes</p>
      </div>
    </div>
  );
}