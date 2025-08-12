import { useState } from 'react';
import type { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const TanstackQueryProvider = ({ children }: PropsWithChildren) => {
  const [client] = useState(new QueryClient({ defaultOptions: { queries: { staleTime: 5000 } } }));

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
};

export default TanstackQueryProvider;
