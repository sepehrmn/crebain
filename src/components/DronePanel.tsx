/**
 * CREBAIN Drone Control Panel
 * UI for spawning, managing, and controlling drones
 */

import { useState, useCallback } from 'react'
import * as THREE from 'three'
import { DRONE_TYPES, type DroneTypeDefinition } from '../physics/DroneTypes'
import type { DronePhysicsBody } from '../physics/DronePhysics'
import { BasePanel } from './BasePanel'
import { CATEGORY_ICONS, CATEGORY_LABELS } from '../lib/droneCategories'

// ─────────────────────────────────────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────────────────────────────────────

type FlightMode = 'manual' | 'stabilized' | 'altitude_hold' | 'position_hold' | 'waypoint'

interface DronePanelProps {
  drones: DronePhysicsBody[]
  selectedDroneId: string | null
  onSpawnDrone: (typeId: string, position: THREE.Vector3) => void
  onRemoveDrone: (id: string) => void
  onSelectDrone: (id: string | null) => void
  onArmDrone: (id: string, armed: boolean) => void
  onSetFlightMode: (id: string, mode: FlightMode) => void
  isExpanded?: boolean
  onToggleExpand?: () => void
}

// ─────────────────────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────────────────────

const FLIGHT_MODE_LABELS: Record<FlightMode, string> = {
  manual: 'MANUELL',
  stabilized: 'STABILISIERT',
  altitude_hold: 'HÖHE HALTEN',
  position_hold: 'POSITION HALTEN',
  waypoint: 'WEGPUNKT',
}

// ─────────────────────────────────────────────────────────────────────────────
// HELPER COMPONENTS
// ─────────────────────────────────────────────────────────────────────────────

function DroneTypeCard({
  droneType,
  onSelect,
  isSelected,
}: {
  droneType: DroneTypeDefinition
  onSelect: () => void
  isSelected: boolean
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full p-2 border text-left transition-colors ${
        isSelected
          ? 'border-[#4a8a5a] bg-[#1a2a1a] text-[#6aba6a]'
          : 'border-[#2a2a2a] bg-[#0a0a0a] text-[#707070] hover:border-[#3a3a3a] hover:bg-[#151515]'
      }`}
      title={`${droneType.description}\n\nMasse: ${droneType.physics.mass}kg\nMax. Geschw.: ${droneType.physics.maxSpeed}m/s\nFlugzeit: ${droneType.physics.endurance}min`}
    >
      <div className="flex items-center gap-2">
        <span className="text-lg">{CATEGORY_ICONS[droneType.category]}</span>
        <div className="flex-1 min-w-0">
          <div className="text-[1.125em] font-bold truncate">{droneType.name}</div>
          <div className="text-[0.875em] text-[#505050]">
            {CATEGORY_LABELS[droneType.category]}
          </div>
        </div>
      </div>
      <div className="mt-1 grid grid-cols-3 gap-1 text-[0.75em] text-[#404040]">
        <div>{droneType.physics.mass}kg</div>
        <div>{droneType.physics.maxSpeed}m/s</div>
        <div>{droneType.physics.endurance}min</div>
      </div>
    </button>
  )
}

function ActiveDroneCard({
  drone,
  isSelected,
  onSelect,
  onRemove,
  onArm,
}: {
  drone: DronePhysicsBody
  isSelected: boolean
  onSelect: () => void
  onRemove: () => void
  onArm: (armed: boolean) => void
}) {
  const { state } = drone
  const altitude = state.position.y.toFixed(1)
  const speed = state.velocity.length().toFixed(1)
  const batteryPercent = Math.round(state.battery * 100)
  
  // Battery color
  const batteryColor = batteryPercent > 50 
    ? '#4a8a4a' 
    : batteryPercent > 20 
      ? '#8a8a4a' 
      : '#8a4a4a'

  return (
    <div
      className={`p-2 border cursor-pointer transition-colors ${
        isSelected
          ? 'border-[#4a8a5a] bg-[#1a2a1a]'
          : 'border-[#2a2a2a] bg-[#0a0a0a] hover:border-[#3a3a3a]'
      }`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="text-[1em] font-bold text-[#808080]">{drone.id}</div>
        <div className="flex gap-1">
          <button
            onClick={(e) => { e.stopPropagation(); onArm(!state.armed); }}
            className={`px-1.5 py-0.5 text-[0.75em] border transition-colors ${
              state.armed
                ? 'border-[#6a4a4a] bg-[#2a1a1a] text-[#aa6a6a]'
                : 'border-[#3a3a3a] bg-[#1a1a1a] text-[#606060]'
            }`}
          >
            {state.armed ? 'ENTWAFFEN' : 'BEWAFFNEN'}
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
            className="px-1.5 py-0.5 text-[0.75em] border border-[#4a2a2a] bg-[#1a0a0a] text-[#8a4a4a] hover:bg-[#2a1a1a]"
          >
            ✕
          </button>
        </div>
      </div>
      
      {/* Telemetry */}
      <div className="grid grid-cols-3 gap-1 text-[0.875em]">
        <div className="text-[#505050]">
          ALT: <span className="text-[#709070]">{altitude}m</span>
        </div>
        <div className="text-[#505050]">
          SPD: <span className="text-[#709070]">{speed}m/s</span>
        </div>
        <div className="text-[#505050]">
          BAT: <span style={{ color: batteryColor }}>{batteryPercent}%</span>
        </div>
      </div>
      
      {/* Position */}
      <div className="mt-1 text-[0.75em] text-[#404040]">
        POS: {state.position.x.toFixed(1)}, {state.position.y.toFixed(1)}, {state.position.z.toFixed(1)}
      </div>
      
      {/* Armed indicator */}
      {state.armed && (
        <div className="mt-1 text-[0.75em] text-[#aa6a6a] animate-pulse">
          ⚠ BEWAFFNET - MOTOREN AKTIV
        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

export function DronePanel({
  drones,
  selectedDroneId,
  onSpawnDrone,
  onRemoveDrone,
  onSelectDrone,
  onArmDrone,
  onSetFlightMode,
  isExpanded = true,
  onToggleExpand,
}: DronePanelProps) {
  const [selectedTypeId, setSelectedTypeId] = useState<string>('maverick')
  const [activeTab, setActiveTab] = useState<'spawn' | 'active' | 'controls'>('spawn')
  const [flightMode, setFlightMode] = useState<FlightMode>('stabilized')
  
  const droneTypes = Object.values(DRONE_TYPES)
  const selectedDrone = drones.find(d => d.id === selectedDroneId)
  
  const handleSpawn = useCallback(() => {
    // Spawn at a random position above the scene
    const position = new THREE.Vector3(
      (Math.random() - 0.5) * 20,
      10 + Math.random() * 10,
      (Math.random() - 0.5) * 20
    )
    onSpawnDrone(selectedTypeId, position)
  }, [selectedTypeId, onSpawnDrone])
  
  const handleFlightModeChange = useCallback((mode: FlightMode) => {
    setFlightMode(mode)
    if (selectedDroneId) {
      onSetFlightMode(selectedDroneId, mode)
    }
  }, [selectedDroneId, onSetFlightMode])

  return (
    <BasePanel
      panelId="drone"
      title="DROHNEN-STEUERUNG"
      icon="🚁"
      isExpanded={isExpanded}
      onToggleExpand={onToggleExpand}
      headerRight={<span className="text-[#505050]">{drones.length} AKTIV</span>}
      collapsedContent="🚁"
    >
      {/* Tabs */}
      <div className="flex border-b border-[#1a1a1a]">
        {(['spawn', 'active', 'controls'] as const).map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-1.5 text-[0.875em] transition-colors ${
              activeTab === tab
                ? 'bg-[#1a1a1a] text-[#909090] border-b-2 border-[#4a6a5a]'
                : 'text-[#505050] hover:bg-[#151515]'
            }`}
          >
            {tab === 'spawn' && 'SPAWN'}
            {tab === 'active' && 'AKTIV'}
            {tab === 'controls' && 'STEUERUNG'}
          </button>
        ))}
      </div>
      
      {/* Content */}
      <div className="p-2 max-h-96 overflow-y-auto">
        {/* Spawn Tab */}
        {activeTab === 'spawn' && (
          <div className="space-y-2">
            <div className="text-[#606060] mb-2">DROHNENTYP WÄHLEN:</div>
            
            <div className="space-y-1">
              {droneTypes.map(type => (
                <DroneTypeCard
                  key={type.id}
                  droneType={type}
                  isSelected={selectedTypeId === type.id}
                  onSelect={() => setSelectedTypeId(type.id)}
                />
              ))}
            </div>
            
            <button
              onClick={handleSpawn}
              className="w-full py-2 mt-2 border border-[#4a6a5a] bg-[#1a2a1a] text-[#6a9a6a] hover:bg-[#2a3a2a] transition-colors"
            >
              + DROHNE SPAWNEN
            </button>
          </div>
        )}
        
        {/* Active Drones Tab */}
        {activeTab === 'active' && (
          <div className="space-y-2">
            {drones.length === 0 ? (
              <div className="text-center text-[#404040] py-4">
                KEINE AKTIVEN DROHNEN
              </div>
            ) : (
              drones.map(drone => (
                <ActiveDroneCard
                  key={drone.id}
                  drone={drone}
                  isSelected={selectedDroneId === drone.id}
                  onSelect={() => onSelectDrone(drone.id)}
                  onRemove={() => onRemoveDrone(drone.id)}
                  onArm={(armed) => onArmDrone(drone.id, armed)}
                />
              ))
            )}
          </div>
        )}
        
        {/* Controls Tab */}
        {activeTab === 'controls' && (
          <div className="space-y-3">
            {/* Flight Mode Selector */}
            <div>
              <div className="text-[#606060] mb-1">FLUGMODUS:</div>
              <div className="grid grid-cols-2 gap-1">
                {(Object.keys(FLIGHT_MODE_LABELS) as FlightMode[]).map(mode => (
                  <button
                    key={mode}
                    onClick={() => handleFlightModeChange(mode)}
                    className={`py-1 text-[0.75em] border transition-colors ${
                      flightMode === mode
                        ? 'border-[#4a6a8a] bg-[#1a2a3a] text-[#6a9aba]'
                        : 'border-[#2a2a2a] bg-[#0a0a0a] text-[#505050] hover:border-[#3a3a3a]'
                    }`}
                  >
                    {FLIGHT_MODE_LABELS[mode]}
                  </button>
                ))}
              </div>
            </div>
            
            {/* Keyboard Controls Guide */}
            <div className="border border-[#1a1a1a] bg-[#0e0e0e] p-2">
              <div className="text-[#606060] mb-2">TASTATURSTEUERUNG:</div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[0.875em]">
                <div className="text-[#505050]">W/S</div>
                <div className="text-[#707070]">Pitch (vor/zurück)</div>
                <div className="text-[#505050]">A/D</div>
                <div className="text-[#707070]">Roll (links/rechts)</div>
                <div className="text-[#505050]">Q/E</div>
                <div className="text-[#707070]">Yaw (drehen)</div>
                <div className="text-[#505050]">SHIFT/CTRL</div>
                <div className="text-[#707070]">Throttle (auf/ab)</div>
                <div className="text-[#505050]">SPACE</div>
                <div className="text-[#707070]">Bewaffnen/Entwaffnen</div>
              </div>
            </div>
            
            {/* Selected Drone Info */}
            {selectedDrone && (
              <div className="border border-[#2a2a2a] bg-[#0a0a0a] p-2">
                <div className="text-[#707070] font-bold mb-1">
                  AUSGEWÄHLT: {selectedDrone.id}
                </div>
                <div className="space-y-1 text-[0.875em]">
                  <div className="flex justify-between">
                    <span className="text-[#505050]">Position:</span>
                    <span className="text-[#709070]">
                      {selectedDrone.state.position.x.toFixed(1)}, 
                      {selectedDrone.state.position.y.toFixed(1)}, 
                      {selectedDrone.state.position.z.toFixed(1)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#505050]">Geschwindigkeit:</span>
                    <span className="text-[#709070]">
                      {selectedDrone.state.velocity.length().toFixed(2)} m/s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#505050]">Batterie:</span>
                    <span className={selectedDrone.state.battery > 0.2 ? 'text-[#709070]' : 'text-[#aa6a6a]'}>
                      {Math.round(selectedDrone.state.battery * 100)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-[#505050]">Status:</span>
                    <span className={selectedDrone.state.armed ? 'text-[#aa6a6a]' : 'text-[#6a6a6a]'}>
                      {selectedDrone.state.armed ? 'BEWAFFNET' : 'SICHER'}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </BasePanel>
  )
}

export default DronePanel
