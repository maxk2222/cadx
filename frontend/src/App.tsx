import { useQuery } from "@tanstack/react-query";
import { Suspense, lazy } from "react";
import { Activity, BarChart3, FolderOpen, Library, Settings, type LucideIcon } from "lucide-react";
import {
  BrowserRouter,
  Link,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import HealthDots from "./components/HealthDots";
import Logo from "./components/Logo";
import { ToastProvider } from "./components/ui";
import { api } from "./lib/api";
import EditPage from "./pages/EditPage";
import JobsPage from "./pages/JobsPage";
import LibraryPage from "./pages/LibraryPage";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import SettingsPage from "./pages/SettingsPage";
import VersionPage from "./pages/VersionPage";

// lazy: keeps recharts (~200KB) out of the initial bundle
const StatsPage = lazy(() => import("./pages/StatsPage"));

function NavItem({
  to,
  icon: Icon,
  label,
  active,
  badge,
}: {
  to: string;
  icon: LucideIcon;
  label: string;
  active: boolean;
  badge?: number;
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
        <span className="absolute right-1.5 top-1/2 flex h-4 min-w-4 -translate-y-1/2 items-center justify-center rounded-full bg-brand-600 px-1 text-[10px] font-semibold text-white md:static md:translate-y-0">
          {badge}
        </span>
      ) : null}
    </Link>
  );
}

function SideNav() {
  const loc = useLocation();
  const activeJobs = useQuery({
    queryKey: ["jobs-active-count"],
    queryFn: () => api.listJobsAll({ active: true, limit: 50 }),
    refetchInterval: 8000,
  });
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
          icon={FolderOpen}
          label="Projects"
          active={loc.pathname === "/"}
        />
        <NavItem
          to="/jobs"
          icon={Activity}
          label="Jobs"
          active={loc.pathname === "/jobs"}
          badge={activeJobs.data?.length}
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
        />
      </nav>
      <div className="mt-auto flex justify-center border-t border-zinc-800/80 py-3 md:justify-start md:px-3">
        <HealthDots />
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <ToastProvider>
        <div className="flex min-h-screen">
          <SideNav />
          <main className="min-w-0 flex-1">
            <Routes>
              <Route path="/" element={<ProjectsPage />} />
              <Route path="/projects/:id" element={<ProjectPage />} />
              <Route path="/versions/:id" element={<VersionPage />} />
              <Route path="/versions/:id/edit" element={<EditPage />} />
              <Route path="/jobs" element={<JobsPage />} />
              <Route
                path="/stats"
                element={
                  <Suspense fallback={null}>
                    <StatsPage />
                  </Suspense>
                }
              />
              <Route path="/library" element={<LibraryPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </ToastProvider>
    </BrowserRouter>
  );
}
