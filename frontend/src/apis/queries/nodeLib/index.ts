import { useMutation, useQuery } from '@tanstack/react-query';
import { getNodes, postWorkflows } from '../../axios/nodeLib';
import type { NodeLibrary } from '../../axios/nodeLib/types';

export const useGetNodes = () => {
  return useQuery<NodeLibrary>({
    queryKey: ['getNodes'],
    queryFn: () => getNodes(),
  });
};

export const usePostWorkflows = () => {
  return useMutation({
    mutationFn: (data: { name: string; nodes: string[] }) => postWorkflows(data),
  });
};
