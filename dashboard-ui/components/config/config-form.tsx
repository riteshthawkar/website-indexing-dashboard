"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useSaveConfig } from "@/lib/hooks/use-configs";
import type { ConfigSchema, ConfigSchemaField } from "@/lib/types";
import { Save, RotateCcw } from "lucide-react";

interface ConfigFormProps {
  name: string;
  data: Record<string, unknown>;
  schema?: ConfigSchema;
}

function stringifyValue(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function ComplexField({
  fieldKey,
  value,
  schema,
  onChange,
}: {
  fieldKey: string;
  value: unknown;
  schema?: ConfigSchemaField;
  onChange: (value: unknown) => void;
}) {
  const [text, setText] = useState(stringifyValue(value));
  const [error, setError] = useState<string | null>(null);

  const fallback = Array.isArray(value) ? "[]" : "{}";

  return (
    <div className="space-y-1">
      <Label>{fieldKey}</Label>
      {schema?.description && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
      <Textarea
        value={text}
        onChange={(e) => {
          const next = e.target.value;
          setText(next);
          try {
            const parsed = JSON.parse(next.trim() || fallback);
            setError(null);
            onChange(parsed);
          } catch {
            setError("Invalid JSON. Fix the value before saving.");
          }
        }}
        rows={Math.max(6, text.split("\n").length)}
        className="font-mono text-xs"
      />
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

function ConfigField({
  fieldKey,
  value,
  schema,
  onChange,
}: {
  fieldKey: string;
  value: unknown;
  schema?: ConfigSchemaField;
  onChange: (value: unknown) => void;
}) {
  const valueType = Array.isArray(value) ? "array" : typeof value;
  const schemaType = schema?.type;

  if (valueType === "object" && value !== null) {
    return (
      <ComplexField
        fieldKey={fieldKey}
        value={value}
        schema={schema}
        onChange={onChange}
      />
    );
  }

  if (schemaType === "array" || Array.isArray(value)) {
    return (
      <ComplexField
        fieldKey={fieldKey}
        value={Array.isArray(value) ? value : []}
        schema={schema}
        onChange={onChange}
      />
    );
  }

  if (schemaType === "bool" || valueType === "boolean") {
    return (
      <div className="flex items-center justify-between rounded-md border p-3">
        <div>
          <Label>{fieldKey}</Label>
          {schema?.description && (
            <p className="text-xs text-muted-foreground">{schema.description}</p>
          )}
        </div>
        <Switch checked={Boolean(value)} onCheckedChange={onChange} />
      </div>
    );
  }

  if (schemaType === "int" || schemaType === "float" || valueType === "number") {
    return (
      <div className="space-y-1">
        <Label>{fieldKey}</Label>
        {schema?.description && (
          <p className="text-xs text-muted-foreground">{schema.description}</p>
        )}
        <Input
          type="number"
          value={typeof value === "number" ? value : ""}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === "") {
              onChange(schemaType === "float" ? 0 : 0);
              return;
            }
            onChange(schemaType === "float" ? parseFloat(raw) : parseInt(raw, 10));
          }}
        />
      </div>
    );
  }

  const stringValue = value == null ? "" : String(value);
  const isLongText = stringValue.includes("\n") || stringValue.length > 100;

  return (
    <div className="space-y-1">
      <Label>{fieldKey}</Label>
      {schema?.description && (
        <p className="text-xs text-muted-foreground">{schema.description}</p>
      )}
      {isLongText ? (
        <Textarea
          value={stringValue}
          onChange={(e) => onChange(e.target.value)}
          rows={Math.max(3, stringValue.split("\n").length)}
        />
      ) : (
        <Input
          value={stringValue}
          onChange={(e) => {
            const next = e.target.value;
            if (schemaType === "NoneType" && next === "") {
              onChange(null);
              return;
            }
            onChange(next);
          }}
        />
      )}
    </div>
  );
}

export function ConfigForm({ name, data, schema }: ConfigFormProps) {
  const [formData, setFormData] = useState<Record<string, unknown>>(data);
  const saveMutation = useSaveConfig();

  useEffect(() => {
    setFormData(data);
  }, [data]);

  const handleChange = (key: string, value: unknown) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>{name}</CardTitle>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setFormData(data)}>
              <RotateCcw className="mr-2 h-4 w-4" />
              Reset
            </Button>
            <Button
              size="sm"
              onClick={() => saveMutation.mutate({ name, data: formData })}
              disabled={saveMutation.isPending}
            >
              <Save className="mr-2 h-4 w-4" />
              Save
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {Object.entries(formData).map(([key, value]) => (
          <ConfigField
            key={`${key}:${stringifyValue(value)}`}
            fieldKey={key}
            value={value}
            schema={schema?.[key]}
            onChange={(nextValue) => handleChange(key, nextValue)}
          />
        ))}
      </CardContent>
    </Card>
  );
}
