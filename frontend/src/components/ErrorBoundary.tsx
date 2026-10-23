import { AlertTriangle, RotateCw } from "lucide-react";
import { Component, type ErrorInfo, type ReactNode } from "react";
import i18n from "../i18n";

interface Props {
  children: ReactNode;
  /** smaller inline fallback instead of the full-page screen (e.g. panels) */
  inline?: boolean;
}
interface State {
  error: Error | null;
}

/** Last-resort error boundary — a render crash anywhere below must not take
 * the whole app down to a white screen. The boundary re-renders on language
 * change via i18n.t() at render time (same pattern as ui.tsx). */
export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // keep the stack in the console — never surfaces to users verbatim
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  private reset = () => {
    this.setState({ error: null });
  };

  render() {
    const err = this.state.error;
    if (!err) return this.props.children;
    if (this.props.inline) {
      return (
        <div className="flex items-center gap-2 rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span className="flex-1">{i18n.t("error_boundary.inline")}</span>
          <button
            type="button"
            onClick={this.reset}
            className="rounded border border-red-400/40 px-2 py-0.5 text-xs hover:bg-red-500/20"
          >
            {i18n.t("action.retry")}
          </button>
        </div>
      );
    }
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-bg p-6 text-center">
        <AlertTriangle className="h-10 w-10 text-amber-400" />
        <h1 className="text-lg font-semibold text-fg">{i18n.t("error_boundary.title")}</h1>
        <p className="max-w-md text-sm text-muted">{i18n.t("error_boundary.hint")}</p>
        <details className="max-w-lg text-left text-xs text-faint">
          <summary className="cursor-pointer">{i18n.t("error_boundary.details")}</summary>
          <pre className="mt-2 max-h-40 overflow-auto rounded border border-edge bg-panel p-2">
            {String(err?.message ?? err)}
          </pre>
        </details>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={this.reset}
            className="flex items-center gap-2 rounded-md border border-edge bg-panel px-4 py-2 text-sm text-fg hover:bg-panel2"
          >
            <RotateCw className="h-4 w-4" />
            {i18n.t("action.retry")}
          </button>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-md bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-500"
          >
            {i18n.t("error_boundary.reload")}
          </button>
        </div>
      </div>
    );
  }
}
