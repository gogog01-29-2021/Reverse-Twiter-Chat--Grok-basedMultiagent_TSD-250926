import colors from './color';
import mediaQuery from './mediaQuery';
import spacing from './spacing';
import typography from './typography';

const theme = {
  mediaQuery,
  spacing,
  typography,
  colors,
} as const;

export type ThemeType = typeof theme;

export default theme;
