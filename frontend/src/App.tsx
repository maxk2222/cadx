import { Box, FolderOpen, Settings } from "lucide-react";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import HealthDots from "./components/HealthDots";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import SettingsPage from "./pages/SettingsPage";
import VersionPage from "./pages/VersionPage";

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen flex-col">
        <nav className="sticky top-0 z-40 flex items-center gap-4 border-b border-zinc-800 bg-zinc-950/85 px-5 py-2.5 backdrop-blur">
          <Link to="/" className="flex items-center gap-2 text-lg font-bold tracking-tight">
            <Box className="h-5 w-5 text-blue-500" />
            cad<span className="text-blue-500">agent</span>
          </Link>
          <Link
            to="/"
            className="flex items-center gap-1.5 text-sm text-zinc-400 hover:text-zinc-200"
          >
            <FolderOpen className="h-4 w-4" />
            projects
          </Link>
          <span className="flex-1" />
          <HealthDots />
          <Link
            to="/settings"
            className="flex items-center gap-1.5 text-sm text-zinc-400 hover:text-zinc-200"
          >
            <Settings className="h-4 w-4" />
            settings
          </Link>
        </nav>
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<ProjectsPage />} />
            <Route path="/projects/:id" element={<ProjectPage />} />
            <Route path="/versions/:id" element={<VersionPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
