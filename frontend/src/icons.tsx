import type { ReactNode } from 'react'

// Small line icons drawn inline, so no icon library is needed. They take the
// current text colour and are hidden from screen readers.
function Icon({ size = 20, children }: { size?: number; children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  )
}

export function BookIcon() {
  return (
    <Icon size={22}>
      <path d="M12 6.5C10 5 7 4.5 3 5v13c4-.5 7 0 9 1.5 2-1.5 5-2 9-1.5V5c-4-.5-7 0-9 1.5z" />
      <path d="M12 6.5v13" />
    </Icon>
  )
}

export function SearchIcon() {
  return (
    <Icon>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </Icon>
  )
}

export function InfoIcon() {
  return (
    <Icon size={22}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <path d="M12 7.5h.01" />
    </Icon>
  )
}

export function AlertIcon() {
  return (
    <Icon size={22}>
      <path d="M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20.5h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
      <path d="M12 9v4" />
      <path d="M12 16.5h.01" />
    </Icon>
  )
}

export function ChevronIcon() {
  return (
    <Icon size={16}>
      <path d="m6 9 6 6 6-6" />
    </Icon>
  )
}
