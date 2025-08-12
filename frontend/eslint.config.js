module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'airbnb',
    'airbnb/hooks',
    'eslint-config-airbnb-typescript',
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react-hooks/recommended',
    'plugin:jsx-a11y/recommended',
    'plugin:import/errors',
    'plugin:import/warnings',
    'prettier',
    'plugin:react-hooks/recommended',
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'vite.config.ts', 'sdk.tsx'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    project: './tsconfig.json',
    ecmaFeatures: {
      jsx: true,
    },
  },
  plugins: [
    'react-refresh',
    '@typescript-eslint',
    'react',
    'react-hooks',
    'jsx-a11y',
    'import',
    'prettier',
  ],
  rules: {
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    //Prettier의 규칙에 맞지 않는 경우 에러를 발생
    'prettier/prettier': ['error'],
    'arrow-body-style': 'off',
    'prefer-arrow-callback': 'off',
    // 함수의 매개변수의 속성에 대한 재할당은 허용
    'no-param-reassign': ['error', { props: false }],
    // import React from 'react' 미사용 error 억제
    'react/react-in-jsx-scope': 'off',
    // 콘솔 사용 시 발생하는 경고 비활성화
    'no-console': ['off'],
    // export문이 하나일 때 default export 사용 권장 경고 비활성화
    'import/prefer-default-export': ['off'],
    // react hooks의 의존성배열이 충분하지 않을 때 경고 표시
    'react-hooks/exhaustive-deps': ['off'],
    // 컴포넌트 이름은 PascalCase로
    'react/jsx-pascal-case': 'warn',
    'react/self-closing-comp': 'warn', // 셀프 클로징 태그 가능하면 적용
    'react/jsx-curly-brace-presence': 'off', // jsx 내 불필요한 중괄호 금지 에러 비활성화
    'react/destructuring-assignment': 'warn', // state, prop 등에 구조분해 할당 적용
    'no-unused-vars': 'warn', // 사용하지 않는 변수 금지
    // import문에서 확장자를 명시가능 하도록 error off
    'import/extensions': ['off'],
    // default props 억제
    'react/require-default-props': 'off',
    'jsx-a11y/click-events-have-key-events': 'off',
    'no-nested-ternary': 'off',
    'react/no-array-index-key': 'off',
    'react/no-unstable-nested-components': 'off',
    // 리액트 훅 폼사용으로 spread 허용
    'react/jsx-props-no-spreading': 'off',
    // 리액트 빈 태그 허용
    'react/jsx-no-useless-fragment': 'off',
    'react/jsx-key': 'off',
    'jsx-a11y/label-has-associated-control': [
      2,
      {
        some: ['nesting', 'id'],
      },
    ],
    '@typescript-eslint/no-unused-vars': 'off',
    'import/no-unresolved': 'off',
    'react/function-component-definition': [
      2,
      { namedComponents: ['arrow-function', 'function-declaration'] },
    ],
  },
  settings: {
    'import/resolver': {
      node: {
        extensions: ['.js', '.jsx', '.ts', '.tsx'],
      },
    },
    react: {
      version: 'detect',
    },
  },
};
