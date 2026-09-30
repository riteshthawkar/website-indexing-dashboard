"use client";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface StatusFilterProps {
  value: string;
  onChange: (value: string) => void;
}

export function StatusFilter({ value, onChange }: StatusFilterProps) {
  return (
    <Select
      value={value}
      items={[
        { value: "all", label: "All statuses" },
        { value: "completed", label: "Completed" },
        { value: "running", label: "Running" },
        { value: "failed", label: "Failed" },
        { value: "pending", label: "Pending" },
        { value: "cancelled", label: "Cancelled" },
      ]}
      onValueChange={(v) => onChange(v || "all")}
    >
      <SelectTrigger
        aria-label="Filter projects by status"
        className="h-9 w-40"
      >
        <SelectValue placeholder="All statuses" />
      </SelectTrigger>
      <SelectContent>
        <SelectGroup>
          <SelectItem value="all">All statuses</SelectItem>
          <SelectItem value="completed">Completed</SelectItem>
          <SelectItem value="running">Running</SelectItem>
          <SelectItem value="failed">Failed</SelectItem>
          <SelectItem value="pending">Pending</SelectItem>
          <SelectItem value="cancelled">Cancelled</SelectItem>
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
