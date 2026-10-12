import { FolderOpen, Library, Settings, type LucideIcon } from "lucide-react";
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
import EditPage from "./pages/EditPage";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import SettingsPage from "./pages/SettingsPage";
import VersionPage from "./pages/VersionPage";

function NavItem({
  to,
  icon: Icon,
  label,
  active,
}: {
  to: string;
  icon: LucideIcon;
  label: string;
  active: boolean;
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
    </Link>
  );
}

function SideNav() {
  const loc = useLocation();
  const tab = new URLSearchParams(loc.search).get("tab");
  const onSettings = loc.pathname === "/settings";
  return (
    <aside className="sticky top-0 z-40 flex h-screen w-14 shrink-0 flex-col border-r border-zinc-800/80 bg-zinc-950 md:w-56">
      <Link
        to="/"
        title="CADX — home"
        className="mx-2 mb-2 mt-3 flex h-11 items-center justify-center rounded-lg transition-colors hover:bg-zinc-900 md:justify-start md:px-2"
      >
        <span className="hidden md:inline">
          <Logo size={24} wordmark />
        </span>
        <span className="md:hidden">
          <Logo size={24} />
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
          to="/settings?tab=library"
          icon={Library}
          label="Library"
          active={onSettings && tab === "library"}
        />
        <NavItem
          to="/settings?tab=provider"
          icon={Settings}
          label="Settings"
          active={onSettings && tab !== "library"}
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
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </div>
      </ToastProvider>
    </BrowserRouter>
  );
}
