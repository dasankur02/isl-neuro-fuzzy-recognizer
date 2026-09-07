import { Activity, BrainCircuit, FlaskConical, Gauge, Hexagon, Menu, Radio, X } from 'lucide-react';
import { useState } from 'react';
import type { ReactNode } from 'react';
import { Link, useLocation } from 'wouter';
import { useHealthCheck, getHealthCheckQueryKey } from '@workspace/api-client-react';

const navigation = [
  { href: '/', label: 'Live workbench', hint: 'recognition', icon: Radio },
  { href: '/training', label: 'Training lab', hint: 'dual-stage fitting', icon: BrainCircuit },
  { href: '/evaluation', label: 'Evaluation', hint: 'model comparison', icon: FlaskConical },
];

function StatusPill() {
  const { data, isLoading, isError } = useHealthCheck({
    query: { queryKey: getHealthCheckQueryKey(), refetchInterval: 30000 },
  });
  const healthy = !isError && data?.status !== 'degraded';
  return (
    <div className="flex items-center gap-2 text-[11px] font-semibold tracking-[0.08em] text-sidebar-foreground/60" data-testid="status-api-health">
      <span className={`relative flex size-2 ${isLoading ? 'bg-sidebar-foreground/40' : healthy ? 'bg-[#f6c341]' : 'bg-[#ee765e]'}`}>
        <span className={`absolute inset-0 animate-ping ${isLoading ? 'bg-sidebar-foreground/20' : healthy ? 'bg-[#f6c341]' : 'bg-[#ee765e]'} opacity-60`} />
      </span>
      {isLoading ? 'CHECKING CORE' : healthy ? 'CORE ONLINE' : 'CORE UNAVAILABLE'}
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-[100dvh] bg-background">
      <header className="fixed inset-x-0 top-0 z-40 flex h-[68px] items-center justify-between border-b border-sidebar-border bg-sidebar px-5 text-sidebar-foreground md:hidden">
        <Link href="/" className="flex items-center gap-3" data-testid="link-mobile-brand">
          <span className="flex size-9 items-center justify-center bg-primary text-primary-foreground"><Hexagon size={20} strokeWidth={2.5} /></span>
          <span className="font-display text-lg font-bold tracking-tight">HASTA<span className="text-primary">/</span>LAB</span>
        </Link>
        <button type="button" onClick={() => setMobileOpen((value) => !value)} className="flex size-10 items-center justify-center border border-sidebar-border text-sidebar-foreground" data-testid="button-toggle-navigation" aria-label="Toggle navigation">
          {mobileOpen ? <X size={19} /> : <Menu size={19} />}
        </button>
      </header>

      <aside className={`fixed inset-y-0 left-0 z-50 flex w-[264px] flex-col bg-sidebar text-sidebar-foreground transition-transform duration-300 md:translate-x-0 ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex h-[92px] items-center border-b border-sidebar-border px-7">
          <Link href="/" className="flex items-center gap-3" onClick={() => setMobileOpen(false)} data-testid="link-brand">
            <span className="flex size-10 items-center justify-center bg-primary text-primary-foreground"><Hexagon size={22} strokeWidth={2.4} /></span>
            <div>
              <div className="font-display text-[21px] font-bold leading-none tracking-[-0.04em]">HASTA<span className="text-primary">/</span>LAB</div>
              <div className="mt-1 font-mono text-[9px] uppercase tracking-[0.22em] text-sidebar-foreground/45">ISL instrument 01</div>
            </div>
          </Link>
        </div>

        <div className="px-5 pt-8">
          <div className="mb-3 px-2 font-mono text-[10px] uppercase tracking-[0.2em] text-sidebar-foreground/35">Workspace</div>
          <nav className="space-y-1" aria-label="Primary navigation">
            {navigation.map(({ href, label, hint, icon: Icon }) => {
              const active = location === href;
              return (
                <Link key={href} href={href} onClick={() => setMobileOpen(false)} className={`group flex items-center gap-3 border-l-2 px-3 py-3 transition-all ${active ? 'border-primary bg-sidebar-accent text-sidebar-accent-foreground' : 'border-transparent text-sidebar-foreground/60 hover:border-sidebar-foreground/30 hover:bg-sidebar-accent/60 hover:text-sidebar-foreground'}`} data-testid={`link-nav-${label.toLowerCase().replaceAll(' ', '-')}`}>
                  <Icon size={18} strokeWidth={active ? 2.2 : 1.8} className={active ? 'text-primary' : 'text-sidebar-foreground/45 group-hover:text-primary'} />
                  <span className="min-w-0">
                    <span className="block text-[13px] font-bold">{label}</span>
                    <span className="block truncate font-mono text-[9px] uppercase tracking-[0.08em] text-sidebar-foreground/35">{hint}</span>
                  </span>
                  {active && <span className="ml-auto size-1.5 bg-primary" />}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="mt-auto border-t border-sidebar-border px-7 py-6">
          <div className="mb-5 flex items-center gap-2 text-sidebar-foreground/35">
            <Activity size={14} />
            <span className="font-mono text-[10px] uppercase tracking-[0.14em]">Signal monitor</span>
          </div>
          <StatusPill />
          <div className="mt-4 grid grid-cols-2 gap-2 font-mono text-[10px] text-sidebar-foreground/40">
            <span>MODEL <b className="font-medium text-sidebar-foreground/70">v0.8.4</b></span>
            <span>LATENCY <b className="font-medium text-sidebar-foreground/70">~42ms</b></span>
          </div>
        </div>
      </aside>

      {mobileOpen && <button className="fixed inset-0 z-40 bg-sidebar/60 md:hidden" onClick={() => setMobileOpen(false)} aria-label="Close navigation" data-testid="button-close-navigation" />}
      <main className="min-h-[100dvh] pt-[68px] md:ml-[264px] md:pt-0">{children}</main>
    </div>
  );
}

export function PageHeader({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col gap-5 border-b border-border px-5 py-7 sm:px-8 lg:flex-row lg:items-end lg:justify-between lg:px-10 lg:py-9">
      <div>
        <div className="mb-3 flex items-center gap-2 font-mono text-[10px] font-medium uppercase tracking-[0.22em] text-muted-foreground"><span className="size-1.5 bg-primary" />{eyebrow}</div>
        <h1 className="font-display text-3xl font-bold tracking-[-0.045em] text-foreground sm:text-[40px]">{title}</h1>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">{description}</p>
      </div>
      {action}
    </div>
  );
}

export function DataState({ type, message, action }: { type: 'loading' | 'error' | 'empty'; message: string; action?: React.ReactNode }) {
  return (
    <div className="flex min-h-[180px] flex-col items-center justify-center border border-dashed border-border bg-card/50 p-8 text-center" data-testid={`state-${type}`}>
      <Gauge className={`mb-3 ${type === 'error' ? 'text-destructive' : 'text-muted-foreground/50'}`} size={24} />
      <p className="text-sm font-semibold text-foreground/80">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}