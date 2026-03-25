"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Check, Copy, Terminal } from "lucide-react";

interface CommandBlockProps {
  label: string;
  command: string;
  description?: string;
}

export function CommandBlock({ label, command, description }: CommandBlockProps) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return undefined;
    const timer = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="rounded-2xl border border-border/70 bg-card/70 p-4 shadow-[0_18px_40px_-28px_rgba(0,0,0,0.7)] backdrop-blur">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <Terminal className="h-4 w-4 text-primary" />
            <p className="text-sm font-medium">{label}</p>
          </div>
          {description && (
            <p className="text-xs text-muted-foreground">{description}</p>
          )}
        </div>
        <Badge variant="outline" className="text-[10px] uppercase tracking-[0.16em]">
          CLI
        </Badge>
      </div>
      <div className="flex items-start gap-3 rounded-xl border border-white/8 bg-black/40 p-3">
        <code className="flex-1 overflow-x-auto whitespace-pre-wrap font-mono text-xs text-primary-foreground/90">
          {command}
        </code>
        <Button size="icon" variant="ghost" onClick={handleCopy} className="shrink-0">
          {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
        </Button>
      </div>
    </div>
  );
}
