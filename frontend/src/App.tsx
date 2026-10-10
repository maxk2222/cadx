import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import ProjectPage from "./pages/ProjectPage";
import ProjectsPage from "./pages/ProjectsPage";
import SettingsPage from "./pages/SettingsPage";
import VersionPage from "./pages/VersionPage";

export default function App() {
  return (
    <BrowserRouter>
      <nav className="flex items-center gap-4 border-b border-zinc-800 px-6 py-3">
        <Link to="/" className="text-lg font-bold tracking-tight">
          cad<span className="text-blue-500">agent</span>
        </Link>
        <span className="flex-1" />
        <Link to="/settings" className="text-sm text-zinc-400 hover:text-zinc-200">
          settings
        </Link>
      </nav>
      <Routes>
        <Route path="/" element={<ProjectsPage />} />
        <Route path="/projects/:id" element={<ProjectPage />} />
        <Route path="/versions/:id" element={<VersionPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </BrowserRouter>
  );
}
