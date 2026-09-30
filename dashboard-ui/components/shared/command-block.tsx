"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Check, Copy } from "lucide-react";
import { toast } from "sonner";

interface CommandBlockProps {
  label: string;
  command: string;
  description?: string;
}

export function CommandBlock({
  label,
  command,
  description,
}: CommandBlockProps) {
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
      toast.error(
        "Could not copy the command. Select and copy the text instead.",
      );
    }
  };

  return (
    <div className="flex min-w-0 flex-col gap-3">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div>
          <div className="mb-1 flex items-center gap-2">
            <p className="text-sm font-medium">{label}</p>
          </div>
          {description && (
            <p className="text-xs text-muted-foreground">{description}</p>
          )}
        </div>
      </div>
      <div className="flex items-start gap-3 rounded-lg border bg-muted p-3">
        <code className="min-w-0 flex-1 overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs leading-6">
          {command}
        </code>
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={handleCopy}
          aria-label={
            copied ? "Command copied" : `Copy ${label.toLowerCase()} command`
          }
        >
          {copied ? <Check /> : <Copy />}
        </Button>
      </div>
    </div>
  );
}
