import type { PropsWithChildren } from 'react';
import StyledPropvider from './Styled';
import TanstackQueryProvider from './TanstackQuery';
import ReactErrorBoundary from './ErrorBoundary';

const MainProvider = ({ children }: PropsWithChildren) => {
  return (
    <StyledPropvider>
      <ReactErrorBoundary isMinor={true}>
        <TanstackQueryProvider>{children}</TanstackQueryProvider>
      </ReactErrorBoundary>
    </StyledPropvider>
  );
};

export default MainProvider;
