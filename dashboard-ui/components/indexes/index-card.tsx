"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { PineconeIndex } from "@/lib/types";

export function IndexCard({ index }: { index: PineconeIndex }) {
  const hasError = !!index.error;
  const namespaceCount = Object.keys(index.namespaces || {}).length;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{index.index_name}</CardTitle>
          <Badge
            variant="outline"
            className={
              hasError
                ? "bg-red-500/15 text-red-400 border-red-500/20"
                : "bg-emerald-500/15 text-emerald-400 border-emerald-500/20"
            }
          >
            {hasError ? "Error" : "Connected"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        {hasError ? (
          <p className="text-sm text-destructive">{index.error}</p>
        ) : (
          <dl className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <dt className="text-muted-foreground">Vectors</dt>
              <dd className="font-medium">{index.vector_count.toLocaleString()}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Dimensions</dt>
              <dd className="font-medium">{index.dimension}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Metric</dt>
              <dd className="font-medium">{index.metric || "\u2014"}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Namespaces</dt>
              <dd className="font-medium">{namespaceCount}</dd>
            </div>
          </dl>
        )}
      </CardContent>
    </Card>
  );
}
