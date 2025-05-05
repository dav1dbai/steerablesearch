import { Space_Grotesk, EB_Garamond, Newsreader } from 'next/font/google';

// Space Grotesk - Similar to Styrene A (Anthropic headings)
export const spaceGrotesk = Space_Grotesk({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
  variable: '--font-space-grotesk',
});

// EB Garamond - Similar to Copernicus (Claude headings)
export const garamond = EB_Garamond({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
  variable: '--font-garamond',
});

// Newsreader - Similar to Tiempos Text (body text)
export const newsreader = Newsreader({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  display: 'swap',
  variable: '--font-newsreader',
});