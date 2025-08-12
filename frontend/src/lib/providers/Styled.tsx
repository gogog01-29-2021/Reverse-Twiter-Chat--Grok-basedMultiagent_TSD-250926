import { ThemeProvider } from 'styled-components';
import GlobalStyle from '../../styles/global';
import type { PropsWithChildren } from 'react';
import theme from '../../styles/theme';

const StyledPropvider = ({ children }: PropsWithChildren) => {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyle />
      {children}
    </ThemeProvider>
  );
};

export default StyledPropvider;
