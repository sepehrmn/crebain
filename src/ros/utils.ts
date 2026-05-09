export function normalizeRosNamespace(namespace: string): string {
  return namespace.trim().replace(/^\/+|\/+$/g, '')
}

export function namespacedRosTopic(namespace: string, suffix: string): string {
  const ns = normalizeRosNamespace(namespace)
  const normalizedSuffix = suffix.replace(/^\/+/, '')
  return ns ? `/${ns}/${normalizedSuffix}` : `/${normalizedSuffix}`
}
