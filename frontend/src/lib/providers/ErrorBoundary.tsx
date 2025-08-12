import type { ReactNode } from 'react';
import { ErrorBoundary } from 'react-error-boundary';

const Fallback = () => {
  return <>error</>;
};
// const Fallback = ({ error, resetErrorBoundary }: FallbackProps) => {
// return <ErrorPage error={error} reset={resetErrorBoundary} />;
// }

const ReactErrorBoundary = ({ children }: { children: ReactNode; isMinor?: boolean }) => {
  return (
    <ErrorBoundary fallbackRender={Fallback} onError={() => {}}>
      {children}
    </ErrorBoundary>
  );
};

export default ReactErrorBoundary;
