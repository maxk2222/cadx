import { useQuery } from "@tanstack/react-query";
import { Suspense, lazy } from "react";
import {
  Activity,
  BarChart3,
  FolderOpen,
  Library,
  MessagesSquare,
  Settings,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import {
  BrowserRouter,
  Link,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import HealthDots from "./components/HealthDots";
import Logo from "./components/Logo";
import { Spinner, ToastProvider } from "./components/ui";
import { api } from "./lib/api";
import { useJobsLive } from "./lib/useJobsLive";
import JobsPage from "./pages/JobsPage";
import LibraryPage from "./pages/LibraryPage";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import SettingsPage from "./pages/SettingsPage";
import StartPage from "./pages/StartPage";

// lazy: keeps the heaviest routes (three.js editors, chat, recharts) out of
// the initial bundle
const ChatPage = lazy(() => import("./pages/ChatPage"));
const EditPage = lazy(() => import("./pages/EditPage"));
const VersionPage = lazy(() => import("./pages/VersionPage"));
const StatsPage = lazy(() => import("./pages/StatsPage"));

const RouteFallback = () => (
  <div className="flex h-screen items-center justify-center" role="status">
    <Spinner className="h-6 w-6 text-zinc-500" />
  </div>
);

function NavItem({
  to,
  icon: Icon,
  label,
  active,
  badge,
  badgeWarn,
  badgeTitle,
}: {
  to: string;
  icon: LucideIcon;
  label: string;
  active: boolean;
  badge?: number;
  /** some badge jobs wait for user input — recolour to draw attention */
  badgeWarn?: boolean;
  /** tooltip override (e.g. "N commits behind origin") */
  badgeTitle?: string;
}) {
  return (
    <Link
      to={to}
      title={label}
      aria-current={active ? "page" : undefined}
      className={`relative mx-2 flex items-center justify-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors md:justify-start ${
        active
          ? "bg-brand-950/70 text-brand-300"
          : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"
      }`}
    >
      {active && (
        <span className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-brand-500" />
      )}
      <Icon className="h-[18px] w-[18px] shrink-0" />
      <span className="hidden md:inline">{label}</span>
      {badge ? (
        <span
          title={badgeTitle ?? (badgeWarn ? "a job is waiting for input" : "active jobs")}
          className={`absolute right-1.5 top-1/2 flex h-4 min-w-4 -translate-y-1/2 items-center justify-center rounded-full px-1 text-[10px] font-semibold text-white md:static md:translate-y-0 ${
            badgeWarn ? "bg-amber-500 text-zinc-950" : "bg-brand-600"
          }`}
        >
          {badge}
        </span>
      ) : null}
    </Link>
  );
}

function SideNav({ jobsLive }: { jobsLive: boolean }) {
  const loc = useLocation();
  // the lifecycle stream pushes the active list — the HTTP query is only the
  // pre-snapshot fill; it polls solely while the stream is down (degraded)
  const activeJobs = useQuery({
    queryKey: ["jobs", "active"],
    queryFn: () => api.listJobsAll({ active: true, limit: 50 }),
    refetchInterval: () => (jobsLive ? false : 15000),
  });
  const waiting = activeJobs.data?.some((j) => j.status === "waiting_for_input") ?? false;
  // cheap: serves the daemon's cached check and only self-refreshes when >6h stale
  const updateCheck = useQuery({
    queryKey: ["update", "check"],
    queryFn: api.updateCheck,
    refetchInterval: 10 * 60_000,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const updatesBehind =
    updateCheck.data?.update_available && updateCheck.data.behind
      ? updateCheck.data.behind
      : undefined;
  return (
    <aside className="sticky top-0 z-40 flex h-screen w-14 shrink-0 flex-col border-r border-zinc-800/80 bg-zinc-950 md:w-44">
      <Link
        to="/"
        title="CADX — home"
        className="mx-2 mb-2 mt-3 flex items-center justify-center rounded-lg py-3 transition-colors hover:bg-zinc-900"
      >
        <span className="hidden md:inline">
          <Logo size={56} wordmark />
        </span>
        <span className="md:hidden">
          <Logo size={40} />
        </span>
      </Link>
      <nav className="flex flex-col gap-1">
        <NavItem
          to="/"
          icon={Sparkles}
          label="New part"
          active={loc.pathname === "/"}
        />
        <NavItem
          to="/chat"
          icon={MessagesSquare}
          label="Chat"
          active={loc.pathname.startsWith("/chat")}
        />
        <NavItem
          to="/projects"
          icon={FolderOpen}
          label="Projects"
          active={loc.pathname === "/projects"}
        />
        <NavItem
          to="/jobs"
          icon={Activity}
          label="Jobs"
          active={loc.pathname === "/jobs"}
          badge={activeJobs.data?.length}
          badgeWarn={waiting}
        />
        <NavItem
          to="/stats"
          icon={BarChart3}
          label="Stats"
          active={loc.pathname === "/stats"}
        />
        <NavItem
          to="/library"
          icon={Library}
          label="Library"
          active={loc.pathname === "/library"}
        />
        <NavItem
          to="/settings"
          icon={Settings}
          label="Settings"
          active={loc.pathname === "/settings"}
          badge={updatesBehind}
          badgeTitle={
            updatesBehind ? `${updatesBehind} update${updatesBehind > 1 ? "s" : ""} available` : undefined
          }
        />
      </nav>
      <div className="mt-auto flex justify-center border-t border-zinc-800/80 py-3 md:justify-start md:px-3">
        <HealthDots />
      </div>
    </aside>
  );
}

function JobsLiveBridge() {
  const connected = useJobsLive();
  return <SideNav jobsLive={connected} />;
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <div className="flex min-h-screen">
          <JobsLiveBridge />
          <main className="min-w-0 flex-1">
            <Suspense fallback={<RouteFallback />}>
              <Routes>
                <Route path="/" element={<StartPage />} />
                <Route path="/projects" element={<ProjectsPage />} />
                <Route path="/projects/:id" element={<ProjectPage />} />
                <Route path="/chat" element={<ChatPage />} />
                <Route path="/chat/:chatId" element={<ChatPage />} />
                <Route path="/versions/:id" element={<VersionPage />} />
                <Route path="/versions/:id/edit" element={<EditPage />} />
                <Route path="/jobs" element={<JobsPage />} />
                <Route path="/stats" element={<StatsPage />} />
                <Route path="/library" element={<LibraryPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </Suspense>
          </main>
        </div>
      </ToastProvider>
    </BrowserRouter>
  );
}
