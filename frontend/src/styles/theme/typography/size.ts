import { css } from 'styled-components';
import weights from './weight';

const headline = {
  h1: css`
    font-size: 2.375rem;
    line-height: 1.21;
    ${weights.bold};
  `,
  h2: css`
    font-size: 2rem;
    line-height: 1.25;
    ${weights.bold};
  `,
  h3: css`
    font-size: 1.5rem;
    line-height: 1.33;
    ${weights.bold};
  `,
  h4: css`
    font-size: 1.25rem;
    line-height: 1.4;
    ${weights.bold};
  `,
  h5: css`
    font-size: 1rem;
    line-height: 1.5;
    ${weights.bold};
  `,
};

const body = {
  regular: {
    sm: css`
      font-size: 0.875rem;
      line-height: 1.5;
      ${weights.regular};
    `,

    base: css`
      font-size: 1rem;
      line-height: 1.5;
      ${weights.regular};
    `,

    lg: css`
      font-size: 1.125rem;
      line-height: 1.5;
      ${weights.regular};
    `,
  },

  medium: {
    sm: css`
      font-size: 0.875rem;
      line-height: 1.5;
      ${weights.medium};
    `,

    base: css`
      font-size: 1rem;
      line-height: 1.5;
      ${weights.medium};
    `,

    lg: css`
      font-size: 1.125rem;
      line-height: 1.5;
      ${weights.medium};
    `,
  },

  strong: {
    sm: css`
      font-size: 0.875rem;
      line-height: 1.5;
      ${weights.semibold};
    `,

    base: css`
      font-size: 1rem;
      line-height: 1.5;
      ${weights.semibold};
    `,

    lg: css`
      font-size: 1.125rem;
      line-height: 1.5;
      ${weights.semibold};
    `,
  },

  decoration: {
    underline: css`
      font-size: 1rem;
      line-height: 1.5;
      ${weights.regular};
      text-decoration-line: underline;
    `,
    strikeThrough: {
      base: css`
        font-size: 1rem;
        line-height: 1.5;
        ${weights.regular};
        text-decoration-line: line-through;
      `,

      sm: css`
      font-size: 0.875rem;
      line-height: 1.5;
      ${weights.regular};
      text-decoration-line: line-through;
    } `,
    },
  },
};

const footnote = {
  description: css`
    font-size: 0.875rem;
    line-height: 1.4;
    ${weights.regular};
  `,
};

const button = {
  lg: css`
    font-size: 1.25rem;
    line-height: 1.2;
    ${weights.medium};
    body.kr * {
      ${weights.bold};
    }
  `,

  base: css`
    font-size: 1rem;
    line-height: 1.2;
    ${weights.medium};
    body.kr * {
      ${weights.bold};
    }
  `,

  sm: css`
    font-size: 0.875rem;
    line-height: 1.2;
    ${weights.medium};
    body.kr * {
      ${weights.bold};
    }
  `,
};

const size = { body, button, headline, footnote };

export default size;
