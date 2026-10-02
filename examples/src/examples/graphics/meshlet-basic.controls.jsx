import {
    BindingTwoWay,
    BooleanInput,
    Label,
    LabelGroup,
    Panel,
    SelectInput,
    SliderInput
} from '@playcanvas/pcui/react';

/**
 * @import { Observer } from '@playcanvas/observer'
 * @import { ReactElement } from 'react'
 */

/**
 * @param {object} props - The props.
 * @param {Observer} props.observer - The observer.
 * @returns {ReactElement} The control panel.
 */
export function Controls({ observer }) {
    return (
        <Panel headerText='Meshlet Basic'>
            <LabelGroup text='Asset'>
                <SelectInput
                    type='string'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.asset' }}
                    options={[
                        { v: 'bunny', t: 'Bunny' },
                        { v: 'zorah', t: 'Zorah chunk 003 (local)' },
                        { v: 'magoffice', t: 'MagOffice (local)' }
                    ]}
                />
            </LabelGroup>
            <LabelGroup text='Error px'>
                <SliderInput
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.threshold' }}
                    min={0.25}
                    max={32}
                    precision={2}
                />
            </LabelGroup>
            <LabelGroup text='Drawn'>
                <Label
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.stats' }}
                    value={observer.get('data.stats')}
                />
            </LabelGroup>
            <LabelGroup text='Occlusion culling'>
                <BooleanInput
                    type='toggle'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.occlusion' }}
                />
            </LabelGroup>
            <LabelGroup text='Visualisation'>
                <SelectInput
                    type='string'
                    binding={new BindingTwoWay()}
                    link={{ observer, path: 'data.visualization' }}
                    options={[
                        { v: 'material', t: 'Material' },
                        { v: 'meshlet', t: 'Meshlet colours' },
                        { v: 'lod', t: 'LOD tier' },
                        { v: 'normals', t: 'World normals (with normal maps)' },
                        { v: 'albedo', t: 'Albedo' },
                        { v: 'metalness', t: 'Metalness' },
                        { v: 'roughness', t: 'Roughness' }
                    ]}
                />
            </LabelGroup>
        </Panel>
    );
}
