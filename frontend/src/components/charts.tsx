/** Thin Recharts wrappers with CADX dark-theme defaults — keeps StatsPage lean. */

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export const CHART_COLORS = [
  "#7c9cf5", // brand
  "#34d399", // emerald
  "#fbbf24", // amber
  "#f87171", // red
  "#c084fc", // purple
  "#2dd4bf", // teal
  "#a1a1aa", // zinc
  "#fb923c", // orange
];

const AXIS_STYLE = { fontSize: 11, fill: "#71717a" };
const TOOLTIP_STYLE = {
  contentStyle: {
    background: "#18181b",
    border: "1px solid #3f3f46",
    borderRadius: 8,
    fontSize: 12,
  },
  labelStyle: { color: "#a1a1aa" },
  itemStyle: { color: "#e4e4e7" },
} as const;

export function StatArea({
  data,
  xKey,
  areas,
  height = 220,
}: {
  data: object[];
  xKey: string;
  areas: { key: string; name: string; color?: string }[];
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey={xKey} tick={AXIS_STYLE} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis tick={AXIS_STYLE} tickLine={false} axisLine={false} allowDecimals={false} />
        <Tooltip {...TOOLTIP_STYLE} />
        {areas.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {areas.map((a, i) => (
          <Area
            key={a.key}
            type="monotone"
            dataKey={a.key}
            name={a.name}
            stackId={areas.length > 1 ? "1" : undefined}
            stroke={a.color ?? CHART_COLORS[i % CHART_COLORS.length]}
            fill={a.color ?? CHART_COLORS[i % CHART_COLORS.length]}
            fillOpacity={0.25}
            strokeWidth={1.5}
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function StatBars({
  data,
  xKey,
  bars,
  height = 220,
  layout = "horizontal",
  colorFor,
}: {
  data: object[];
  xKey: string;
  bars: { key: string; name: string; color?: string }[];
  height?: number;
  /** "vertical" = category axis on Y (long labels). */
  layout?: "horizontal" | "vertical";
  /** Per-category color override, e.g. semantic score-bucket colors. */
  colorFor?: (name: string) => string;
}) {
  const vertical = layout === "vertical";
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data}
        layout={layout}
        margin={{ top: 4, right: 8, bottom: 0, left: vertical ? 8 : -18 }}
      >
        <CartesianGrid
          stroke="#27272a"
          strokeDasharray="3 3"
          horizontal={!vertical}
          vertical={vertical}
        />
        {vertical ? (
          <>
            <XAxis type="number" tick={AXIS_STYLE} tickLine={false} axisLine={false} />
            <YAxis
              type="category"
              dataKey={xKey}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={false}
              width={110}
            />
          </>
        ) : (
          <>
            <XAxis
              dataKey={xKey}
              tick={AXIS_STYLE}
              tickLine={false}
              axisLine={false}
              minTickGap={24}
            />
            <YAxis tick={AXIS_STYLE} tickLine={false} axisLine={false} allowDecimals={false} />
          </>
        )}
        <Tooltip {...TOOLTIP_STYLE} />
        {bars.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {bars.map((b, i) => (
          <Bar
            key={b.key}
            dataKey={b.key}
            name={b.name}
            stackId={bars.length > 1 ? "1" : undefined}
            fill={b.color ?? CHART_COLORS[i % CHART_COLORS.length]}
            radius={[3, 3, 0, 0]}
          >
            {colorFor &&
              data.map((row, ri) => (
                <Cell
                  key={ri}
                  fill={colorFor(String((row as Record<string, unknown>)[xKey]))}
                />
              ))}
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

/** Stacked bars + a line on a secondary axis — e.g. outcomes/day + avg score. */
export function StatComposed({
  data,
  xKey,
  bars,
  line,
  height = 220,
}: {
  data: object[];
  xKey: string;
  bars: { key: string; name: string; color: string }[];
  line?: { key: string; name: string; color?: string };
  height?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: -18 }}>
        <CartesianGrid stroke="#27272a" strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey={xKey} tick={AXIS_STYLE} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis yAxisId="l" tick={AXIS_STYLE} tickLine={false} axisLine={false} allowDecimals={false} />
        {line && (
          <YAxis
            yAxisId="r"
            orientation="right"
            domain={[0, 100]}
            tick={AXIS_STYLE}
            tickLine={false}
            axisLine={false}
            width={28}
          />
        )}
        <Tooltip {...TOOLTIP_STYLE} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {bars.map((b) => (
          <Bar
            key={b.key}
            yAxisId="l"
            dataKey={b.key}
            name={b.name}
            stackId="1"
            fill={b.color}
            radius={[0, 0, 0, 0]}
          />
        ))}
        {line && (
          <Line
            yAxisId="r"
            type="monotone"
            dataKey={line.key}
            name={line.name}
            stroke={line.color ?? "#fbbf24"}
            strokeWidth={1.5}
            dot={false}
            connectNulls
          />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function StatDonut({
  data,
  height = 200,
  colorFor,
}: {
  data: { name: string; count: number }[];
  height?: number;
  /** Name-based color override — keeps e.g. "failed" red in any order. */
  colorFor?: (name: string) => string;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="count"
          nameKey="name"
          innerRadius="55%"
          outerRadius="85%"
          paddingAngle={2}
          stroke="none"
        >
          {data.map((d, i) => (
            <Cell
              key={i}
              fill={colorFor?.(d.name) ?? CHART_COLORS[i % CHART_COLORS.length]}
            />
          ))}
        </Pie>
        <Tooltip {...TOOLTIP_STYLE} />
        <Legend
          wrapperStyle={{ fontSize: 11 }}
          formatter={(v: string) => <span className="text-zinc-400">{v}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}
