/**
 * CREBAIN Drone Category Utilities
 * Adaptive Response & Awareness System (ARAS)
 *
 * Shared utilities for drone category display (icons, colors, labels)
 */

import type { DroneCategory } from '../physics/DroneTypes'

// ─────────────────────────────────────────────────────────────────────────────
// ICONS
// ─────────────────────────────────────────────────────────────────────────────

/** Emoji icons for each drone category */
export const CATEGORY_ICONS: Record<DroneCategory, string> = {
  quadcopter: '🚁',
  hexacopter: '⬡',
  fixed_wing: '✈️',
  loitering_munition: '💥',
  vtol: '🛩️',
}

/** Get icon for a drone category with fallback */
export function getCategoryIcon(category: DroneCategory | string): string {
  return CATEGORY_ICONS[category as DroneCategory] ?? '🔹'
}

// ─────────────────────────────────────────────────────────────────────────────
// COLORS
// ─────────────────────────────────────────────────────────────────────────────

/** Hex colors for each drone category */
export const CATEGORY_COLORS: Record<DroneCategory, string> = {
  quadcopter: '#4a9aff',
  hexacopter: '#9a4aff',
  fixed_wing: '#4aff9a',
  loitering_munition: '#ff4a4a',
  vtol: '#ffaa4a',
}

/** Get color for a drone category with fallback */
export function getCategoryColor(category: DroneCategory | string): string {
  return CATEGORY_COLORS[category as DroneCategory] ?? '#888888'
}

// ─────────────────────────────────────────────────────────────────────────────
// LABELS
// ─────────────────────────────────────────────────────────────────────────────

/** German labels for each drone category */
export const CATEGORY_LABELS: Record<DroneCategory, string> = {
  quadcopter: 'QUADCOPTER',
  hexacopter: 'HEXACOPTER',
  fixed_wing: 'STARRFLÜGLER',
  loitering_munition: 'LOITERING MUNITION',
  vtol: 'VTOL',
}

/** Get label for a drone category with fallback */
export function getCategoryLabel(category: DroneCategory | string): string {
  return CATEGORY_LABELS[category as DroneCategory] ?? category.toUpperCase()
}
