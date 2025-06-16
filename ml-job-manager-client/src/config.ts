// if(!import.meta.env.VITE_BASE_API_JOB) {
//     throw new Error("BASE_JOB_API is required");
// }

const config = {
    // baseJobApi: import.meta.env.VITE_BASE_API_JOB
    baseJobApi: 'http://localhost:10000/api/jobs',
    baseFileApi: 'http://localhost:10000/api/files',
    baseLabellingApi: 'http://localhost:10000/api/labellings',
    baseSpectraApi: 'http://localhost:10000/api/spectra',
};

export default config;