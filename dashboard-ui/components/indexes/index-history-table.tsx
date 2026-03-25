"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { formatDate } from "@/lib/utils";
import type { PineconeSnapshot } from "@/lib/types";

export function IndexHistoryTable({ snapshots }: { snapshots: PineconeSnapshot[] }) {
  if (snapshots.length === 0) {
    return <p className="py-4 text-center text-sm text-muted-foreground">No history available.</p>;
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Captured At</TableHead>
          <TableHead className="text-right">Vectors</TableHead>
          <TableHead className="text-right">Dimensions</TableHead>
          <TableHead>Metric</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {snapshots.map((snap) => (
          <TableRow key={snap.id}>
            <TableCell>{formatDate(snap.captured_at)}</TableCell>
            <TableCell className="text-right">{snap.vector_count.toLocaleString()}</TableCell>
            <TableCell className="text-right">{snap.dimension}</TableCell>
            <TableCell>{snap.metric || "\u2014"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
