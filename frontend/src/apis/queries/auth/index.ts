import { useMutation } from '@tanstack/react-query';
import { getToken } from '../../axios/auth';

export const useGetToken = () => {
  return useMutation({
    mutationFn: async () => {
      const storedToken = localStorage.getItem('authToken');
      if (storedToken) {
        try {
          const parsed = JSON.parse(storedToken);
          console.log('Using cached token');
          return parsed;
        } catch (error) {
          localStorage.removeItem('authToken');
        }
      }

      console.log('Fetching new token');
      const response = await getToken();

      localStorage.setItem('authToken', JSON.stringify(response));

      return response;
    },
  });
};
