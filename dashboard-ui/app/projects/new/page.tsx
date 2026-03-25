"use client";

import { PageHeader } from "@/components/layout/page-header";
import { CommandBlock } from "@/components/shared/command-block";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { NewProjectForm } from "@/components/new-project/new-project-form";

export default function NewProjectPage() {
  return (
    <>
      <PageHeader title="New Project" description="Create a new pipeline run" />
      <div className="page-section grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_380px]">
        <NewProjectForm />
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>CLI Equivalent</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <CommandBlock
                label="Run from terminal"
                command="./scripts/pipeline.sh run --config default"
                description="Use the shared backend directly without opening the dashboard."
              />
              <CommandBlock
                label="Resume a failed run"
                command="./scripts/pipeline.sh run --config default --resume"
                description="Resume from the last saved checkpoint in the run directory."
              />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Execution Notes</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p>Dashboard-created runs and terminal-created runs land in the same `runs/` workspace.</p>
              <p>Use a descriptive project name. It becomes the operational label in the dashboard and logs.</p>
              <p>If you run the pipeline from the terminal first, use the dashboard scan action to import it.</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </>
  );
}
