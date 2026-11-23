// Where the backend's /ask endpoint lives. Change this if the backend runs elsewhere.
export const ASK_URL = 'http://localhost:8000/ask'

// One-click questions shown under the search box. The last one is not covered
// by the documents, so it shows what an "I don't know" answer looks like.
export const EXAMPLE_QUESTIONS = [
  'What torque for the pump flange bolts?',
  'Which oil grade should I use?',
  'What does part HX40-SEAL-28S-316 cost?',
  'What is the warranty?',
]

// Short names for the source chips. A file not listed here is shown by its
// file name, without the number prefix and ".pdf".
export const DOCUMENT_NAMES: Record<string, string> = {
  '01_HX-400_IOM_Manual.pdf': 'HX-400 Manual',
  '02_RIS-VGM-0042_Safety_Register.pdf': 'Safety Register',
  '03_WP-114_Isolation_and_Confined_Space.pdf': 'WP-114 Isolation',
  '04_HX-SPC-2026_Parts_Catalogue.pdf': 'Parts Catalogue',
  '05_OV-220_Preventief_Onderhoud_NL.pdf': 'OV-220 Maintenance (NL)',
}
