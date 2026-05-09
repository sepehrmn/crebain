export const TRANSPORT_EVENT_PREFIX = 'crebain.transport.'

export function getTransportEventName(topic: string): string {
  const bytes = new TextEncoder().encode(topic)
  let encoded = TRANSPORT_EVENT_PREFIX

  for (const byte of bytes) {
    const char = String.fromCharCode(byte)
    encoded += /^[A-Za-z0-9_.-]$/.test(char)
      ? char
      : `%${byte.toString(16).toUpperCase().padStart(2, '0')}`
  }

  return encoded
}
